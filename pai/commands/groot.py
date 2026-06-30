"""pai groot — Lab 1 GR00T workflow: convert data, train, deploy, serve.

Each subcommand is a thin wrapper over the validated Lab 1 scripts under
training/groot/ — the CLI resolves Foundation stack outputs and forwards to the
same code the lab doc runs by hand, so there is one source of truth:

    convert   zarr teleop episodes      -> LeRobot v2 dataset  (local)
    launch    LeRobot dataset           -> SageMaker Pipeline run (registers to groot-models)
    runs      list recent pipeline executions
    deploy    a trained model.tar.gz    -> SageMaker real-time endpoint
    invoke    call a live endpoint with a wrist image + robot state
    delete    tear down the endpoint (+ its config + model)
    ingest    bring-your-own Zarr data: convert -> upload -> [train]
"""

import os
import subprocess
import sys
from pathlib import Path

import click

from pai import cfn, config, helpers

STACK_NAME = "PhysicalAi-dev-Foundation"


@click.group()
def groot():
    """Lab 1 — fine-tune and deploy a GR00T manipulation policy."""
    pass


def _ensure_repo_on_path():
    """Put the repo root on sys.path so `import training...` resolves.

    The training/ tree ships with the repo but is not part of the installed
    `pai` package, so commands that import training.groot.* must add the repo
    root first (mirrors rl.py, which does the same for training.scripts.*).
    """
    repo = str(config.REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def _require_data_deps():
    """Abort with a clear message if the data-conversion deps are missing.

    The convert/ingest scripts need zarr + opencv + pandas + pyarrow. These are
    part of the base install, so this only fails if `pai` was installed without
    its dependencies — catch it here and point at the install, instead of letting
    the subprocess die on a raw `ModuleNotFoundError` traceback.
    """
    import importlib.util

    missing = [m for m in ("zarr", "cv2", "pandas", "pyarrow")
               if importlib.util.find_spec(m) is None]
    if missing:
        helpers.error("Missing data-conversion dependencies: " + ", ".join(missing))
        helpers.info("Reinstall `pai` with its dependencies:")
        helpers.info("  pip install -e .")
        raise click.Abort()


def _groot_script(name: str) -> str:
    """Absolute path to a training/groot/<name> script."""
    return str(config.REPO_ROOT / "training" / "groot" / name)


# ---------------------------------------------------------------------------
# convert — Zarr teleop episodes -> LeRobot v2 dataset (local, no AWS)
# ---------------------------------------------------------------------------


@groot.command()
@click.option("--episodes-dir", default="training/data/episodes/episodes",
              help="Directory of raw Zarr episode_* folders")
@click.option("--output-dir", default="training/data/ur3_lerobot_dataset",
              help="Destination LeRobot v2 dataset directory")
@click.option("--camera", default="wrist", help="Primary camera key in the Zarr images group")
@click.option("--dry-run", is_flag=True, help="Print the command without converting")
def convert(episodes_dir, output_dir, camera, dry_run):
    """Convert Zarr teleop episodes to a LeRobot v2 dataset (Lab 1 Step 2)."""
    script = _groot_script("convert_zarr_to_lerobot.py")
    cmd = [sys.executable, script,
           "--episodes-dir", episodes_dir,
           "--output-dir", output_dir,
           "--camera", camera]

    if dry_run:
        helpers.info("[dry-run] Would run:")
        helpers.info(f"  {' '.join(cmd)}")
        helpers.info("\n[dry-run] No data converted.")
        return

    src = Path(episodes_dir)
    if not src.is_dir():
        helpers.error(f"Episodes directory not found: {episodes_dir}")
        helpers.info("Pull and extract the bundled dataset first (Lab 1 Step 2b):")
        helpers.info("  git lfs pull && unzip -o training/data/ur3_episodes_001_027.zip "
                     "-d training/data/episodes")
        raise click.Abort()

    _require_data_deps()

    helpers.heading("Converting Zarr -> LeRobot v2")
    try:
        helpers.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        helpers.error(f"Conversion failed with exit code {e.returncode}")
        raise click.Abort()


# ---------------------------------------------------------------------------
# upload — sync the converted LeRobot dataset to S3 (Lab 1 Step 3)
# ---------------------------------------------------------------------------


@groot.command()
@click.option("--dataset-dir", default="training/data/ur3_lerobot_dataset",
              help="Local LeRobot dataset directory to upload")
@click.option("--prefix", default="groot-data/ur3",
              help="S3 prefix under the datasets bucket (uploads to <prefix>/dataset/)")
@click.option("--dry-run", is_flag=True, help="Resolve the bucket and print the sync; upload nothing")
def upload(dataset_dir, prefix, dry_run):
    """Upload the converted LeRobot dataset to S3 (Lab 1 Step 3).

    Resolves the datasets bucket from the Foundation stack, then runs
    `aws s3 sync` to s3://<bucket>/<prefix>/dataset/ — where `pai groot launch`
    expects to find it.
    """
    region = config.resolve_region()

    bucket = cfn.bucket(STACK_NAME, region)
    if not bucket:
        helpers.error("Could not resolve the datasets bucket from the Foundation stack.")
        helpers.info("Deploy it first: pai deploy foundation")
        raise click.Abort()

    s3_uri = f"s3://{bucket}/{prefix}/dataset/"
    cmd = ["aws", "s3", "sync", dataset_dir, s3_uri]

    if dry_run:
        helpers.info("[dry-run] Would run:")
        helpers.info(f"  {' '.join(cmd)}")
        helpers.info("\n[dry-run] No upload performed.")
        return

    src = Path(dataset_dir)
    if not src.is_dir():
        helpers.error(f"Dataset directory not found: {dataset_dir}")
        helpers.info("Convert the episodes first: pai groot convert")
        raise click.Abort()

    helpers.heading(f"Uploading dataset to {s3_uri}")
    try:
        helpers.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        helpers.error(f"Upload failed with exit code {e.returncode}")
        raise click.Abort()
    helpers.success("\nUpload complete. Launch training with: pai groot launch")


# ---------------------------------------------------------------------------
# launch — drive the SageMaker Pipeline (create-if-missing -> execute)
# ---------------------------------------------------------------------------


@groot.command()
@click.option("--dataset-prefix", default="groot-data/ur3",
              help="S3 prefix under the datasets bucket (expects <prefix>/dataset/)")
@click.option("--max-steps", type=int, default=5000, help="Training steps (100=smoke, 5000=full)")
@click.option("--batch-size", type=int, default=8, help="Per-GPU batch size")
@click.option("--instance-type", default="ml.g5.12xlarge", help="Training instance type")
@click.option("--dry-run", is_flag=True, help="Resolve config and preview; make no AWS calls")
def launch(dataset_prefix, max_steps, batch_size, instance_type, dry_run):
    """Run the GR00T fine-tuning Pipeline on SageMaker (Lab 1 Step 5).

    Creates the pipeline if it does not exist, then starts an execution that
    trains and registers the model to the `groot-models` registry. Upload the
    dataset to s3://<bucket>/<prefix>/dataset/ first (Lab 1 Step 3).
    """
    region = config.resolve_region()

    helpers.heading("GR00T Fine-Tuning Pipeline (SageMaker)")

    helpers.info("[1/2] Reading Foundation stack outputs...")
    outputs = cfn.stack_outputs(STACK_NAME, region)
    bucket = outputs.get("DatasetsBucketName", "")
    role_arn = outputs.get("SageMakerRoleArn", "")
    ecr_uri = outputs.get("GrootTrainingRepoUri", "")
    if not (bucket and role_arn and ecr_uri):
        helpers.error("Failed to read Foundation stack outputs. Is the stack deployed?")
        helpers.info(f"  Bucket:   {bucket or 'MISSING'}")
        helpers.info(f"  Role:     {role_arn or 'MISSING'}")
        helpers.info(f"  ECR:      {ecr_uri or 'MISSING'}")
        helpers.info("\nDeploy it first: pai deploy foundation")
        raise click.Abort()

    image = f"{ecr_uri}:latest"
    helpers.info(f"  Bucket:   {bucket}")
    helpers.info(f"  Role:     {role_arn}")
    helpers.info(f"  Image:    {image}")

    # HF_TOKEN is forwarded to the pipeline at execute time; never echo it.
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        helpers.info("  HF_TOKEN: <redacted> (passed to the pipeline at execute time)")
    else:
        helpers.warn("  HF_TOKEN: not set — base-model download runs unauthenticated (rate-limit-prone)")

    if dry_run:
        helpers.info("\n[2/2] [dry-run] Would create/update pipeline 'groot-finetune-pipeline' then execute:")
        helpers.info(f"  dataset:     s3://{bucket}/{dataset_prefix}/dataset/")
        helpers.info(f"  max-steps:   {max_steps}")
        helpers.info(f"  batch-size:  {batch_size}")
        helpers.info(f"  instance:    {instance_type}")
        helpers.info(f"  registry:    groot-models")
        helpers.info("\n[dry-run] No AWS calls made.")
        return

    _ensure_repo_on_path()
    from training.groot import pipeline

    helpers.info("\n[2/2] Creating/updating pipeline and starting execution...")
    try:
        pipeline.create_pipeline(
            s3_bucket=bucket, role_arn=role_arn, ecr_image=image,
            region=region, instance_type=instance_type,
        )
        result = pipeline.execute_pipeline(
            dataset_prefix=dataset_prefix, max_steps=max_steps,
            batch_size=batch_size, region=region,
        )
    except Exception as e:
        helpers.friendly_boto_error(e)
        return

    if result.get("status") == "error":
        helpers.error(result["message"])
        raise click.Abort()

    helpers.success("\nPipeline execution started.")
    helpers.info(f"  ARN: {result.get('execution_arn', '')}")
    helpers.info("\nMonitor with:")
    helpers.info("  pai groot runs")


# ---------------------------------------------------------------------------
# runs — list recent pipeline executions
# ---------------------------------------------------------------------------


@groot.command()
def runs():
    """List recent GR00T pipeline executions (Lab 1 Step 7b)."""
    region = config.resolve_region()

    _ensure_repo_on_path()
    from training.groot import pipeline

    try:
        result = pipeline.list_runs(region=region)
    except Exception as e:
        helpers.friendly_boto_error(e)
        return

    if result.get("status") == "error":
        helpers.error(result["message"])
        return

    helpers.heading(f"Pipeline: {result['pipeline']}")
    runs_list = result.get("runs", [])
    if not runs_list:
        helpers.info("No executions yet. Start one with: pai groot launch")
        return
    for r in runs_list:
        helpers.info(f"  {r['created']}  {r['status']:<12}  {r['arn']}")


# ---------------------------------------------------------------------------
# deploy / invoke / delete — SageMaker real-time endpoint (Lab 1 Step 9)
# ---------------------------------------------------------------------------


@groot.command()
@click.option("--model-s3", required=True, help="S3 URI of a training job's model.tar.gz")
@click.option("--endpoint-name", default="groot-ur3", help="Endpoint name to create")
@click.option("--instance-type", default="ml.g5.2xlarge", help="Endpoint instance type (needs a GPU)")
@click.option("--dry-run", is_flag=True, help="Show the API calls; make none")
def deploy(model_s3, endpoint_name, instance_type, dry_run):
    """Serve a fine-tuned model as a SageMaker real-time endpoint (Lab 1 Step 9).

    The endpoint runs the groot-inference container and loads model.tar.gz.
    GR00T loads slowly, so the startup health-check timeout is 30 minutes.
    """
    region = config.resolve_region()

    _ensure_repo_on_path()
    from training.groot import deploy_endpoint

    try:
        result = deploy_endpoint.deploy(
            model_s3=model_s3, endpoint_name=endpoint_name,
            instance_type=instance_type, region=region, dry_run=dry_run,
        )
    except Exception as e:
        helpers.friendly_boto_error(e)
        return

    if not dry_run:
        helpers.success(f"\nEndpoint '{endpoint_name}' is creating (~10-30 min to come InService).")
        helpers.info("  Call it:  pai groot invoke --endpoint-name "
                     f"{endpoint_name} --image-path wrist.jpg --task '...'")
        helpers.info(f"  Tear down: pai groot delete --endpoint-name {endpoint_name}")


@groot.command()
@click.option("--endpoint-name", default="groot-ur3", help="Endpoint to call")
@click.option("--image-path", required=True, help="Path to a wrist image file (jpg/png)")
@click.option("--state", default="0,0,0,0,0,0,0", help="Comma-separated 7D robot state")
@click.option("--task", default="", help="Task description, e.g. 'pick up the red cube'")
def invoke(endpoint_name, image_path, state, task):
    """Call a live endpoint with a wrist image + robot state -> action chunk."""
    region = config.resolve_region()

    if not Path(image_path).is_file():
        helpers.error(f"Image not found: {image_path}")
        raise click.Abort()

    _ensure_repo_on_path()
    from training.groot import deploy_endpoint

    try:
        deploy_endpoint.invoke(endpoint_name, image_path, state, task, region)
    except Exception as e:
        helpers.friendly_boto_error(e)


@groot.command()
@click.option("--endpoint-name", default="groot-ur3", help="Endpoint to delete")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
def delete(endpoint_name, yes):
    """Delete the endpoint, its config, and its model (stops hourly billing)."""
    region = config.resolve_region()

    if not yes:
        if not click.confirm(f"Delete endpoint '{endpoint_name}' (+ its config and model)?"):
            helpers.info("Aborted.")
            return

    _ensure_repo_on_path()
    from training.groot import deploy_endpoint

    helpers.heading(f"Deleting endpoint {endpoint_name}")
    try:
        deploy_endpoint.delete(endpoint_name, region)
        helpers.success("Done.")
    except Exception as e:
        helpers.friendly_boto_error(e)


# ---------------------------------------------------------------------------
# ingest — bring-your-own Zarr data: convert -> upload -> [train]
# ---------------------------------------------------------------------------


@groot.command()
@click.option("--episodes-dir", required=True, help="Directory of your raw Zarr episode_* folders")
@click.option("--prefix", default="groot-data/custom", help="S3 prefix under the datasets bucket")
@click.option("--train", is_flag=True, help="Launch a training run after upload")
@click.option("--max-steps", type=int, default=100, help="Training steps if --train is set")
@click.option("--dry-run", is_flag=True, help="Show the plan; make no AWS calls and convert nothing")
def ingest(episodes_dir, prefix, train, max_steps, dry_run):
    """Bring your own data: convert Zarr -> upload to S3 -> optionally train.

    The expected Zarr schema is documented in docs/ZARR_SCHEMA.md.
    """
    region = config.resolve_region()
    script = _groot_script("ingest_customer_data.py")
    cmd = [sys.executable, script,
           "--episodes-dir", episodes_dir, "--prefix", prefix,
           "--region", region, "--max-steps", str(max_steps)]
    if train:
        cmd.append("--train")
    if dry_run:
        cmd.append("--dry-run")

    if dry_run:
        helpers.info("[dry-run] Would run:")
        helpers.info(f"  {' '.join(cmd)}")

    if not dry_run:
        if not Path(episodes_dir).is_dir():
            helpers.error(f"Episodes directory not found: {episodes_dir}")
            raise click.Abort()
        _require_data_deps()

    helpers.heading("Ingesting custom data (convert -> upload" + (" -> train" if train else "") + ")")
    try:
        helpers.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        helpers.error(f"Ingestion failed with exit code {e.returncode}")
        raise click.Abort()


def register(cli: click.Group):
    """Register groot commands with the CLI."""
    cli.add_command(groot)
