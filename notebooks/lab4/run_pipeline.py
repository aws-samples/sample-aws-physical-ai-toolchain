#!/usr/bin/env python3
"""
Lab 4 end-to-end pipeline runner.

Runs: build container → RL training → render video → download MP4
Each step polls until complete before moving to the next.
Errors are printed clearly. Fix the underlying files and re-run.

Usage:
    # Full pipeline (rebuild container + train + video):
    python run_pipeline.py --rebuild

    # Skip build, run training + video (default):
    python run_pipeline.py

    # Skip build AND training (re-render video from last checkpoint):
    python run_pipeline.py --skip-training --checkpoint-s3 s3://bucket/path/model.tar.gz

    # Skip build, run training only (no video):
    python run_pipeline.py --skip-video

Flags:
    --rebuild          Force a full CodeBuild container rebuild (~60-90 min)
    --skip-training    Skip RL training (requires --checkpoint-s3)
    --skip-video       Stop after training, don't render video
    --checkpoint-s3    S3 URI of model.tar.gz to use for video (overrides training output)
    --max-iterations   RL training iterations (default: 100)
    --num-envs         Parallel environments (default: 4096)
    --task             Isaac Lab task name (default: PickAndPlaceUR3-v0)
    --instance-type    SageMaker instance type (default: ml.g5.12xlarge)

State file:
    notebooks/lab4/pipeline_state.json — saves job names across re-runs so you can
    resume a partially-completed pipeline without re-running earlier steps.
"""

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import boto3

# ── Constants ──────────────────────────────────────────────────────────────────
REPO_ROOT     = Path(__file__).resolve().parents[1]
LAB4_DIR      = REPO_ROOT / "notebooks" / "lab4"
CONTAINER_DIR = LAB4_DIR / "container"
STATE_FILE    = LAB4_DIR / "pipeline_state.json"

REGION      = boto3.session.Session().region_name or "us-west-2"
ACCOUNT     = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
BUCKET      = f"sagemaker-{REGION}-{ACCOUNT}"
ROLE_ARN    = f"arn:aws:iam::{ACCOUNT}:role/service-role/YOUR_SAGEMAKER_EXECUTION_ROLE"
ECR_REPO    = "physical-ai/isaac-lab"
ECR_URI     = f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/{ECR_REPO}:latest"
CB_PROJECT  = "physical-ai-isaac-lab-build"
S3_ZIP_KEY  = "codebuild/isaac-lab-context.zip"

ecr = boto3.client("ecr",         region_name=REGION)
cb  = boto3.client("codebuild",   region_name=REGION)
sm  = boto3.client("sagemaker",   region_name=REGION)
s3  = boto3.client("s3",          region_name=REGION)
logs = boto3.client("logs",       region_name=REGION)


# ── Helpers ────────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def poll(description: str, check_fn, interval: int = 30, timeout: int = 7200):
    """Poll check_fn() until it returns a truthy value or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = check_fn()
        if result:
            return result
        elapsed = int(time.time() + timeout - deadline) if timeout else 0
        log(f"  {description} — waiting {interval}s ...")
        time.sleep(interval)
    raise TimeoutError(f"Timed out waiting for: {description}")


def get_cw_log_tail(log_group: str, stream_prefix: str, lines: int = 30) -> str:
    """Return the last N lines from a CloudWatch log stream."""
    try:
        streams = logs.describe_log_streams(
            logGroupName=log_group,
            logStreamNamePrefix=stream_prefix,
        )["logStreams"]
        if not streams:
            return "(no log stream found)"
        stream = streams[0]["logStreamName"]
        events = logs.get_log_events(
            logGroupName=log_group,
            logStreamName=stream,
            limit=lines,
            startFromHead=False,
        )["events"]
        return "\n".join(e["message"].strip() for e in events if e["message"].strip())
    except Exception as e:
        return f"(could not fetch logs: {e})"


# ── Step 1: Build container ────────────────────────────────────────────────────

def build_zip():
    """Package the build context zip and upload to S3."""
    log("Packaging build context zip ...")

    BUILDSPEC = f"""version: 0.2
phases:
  pre_build:
    commands:
      - echo Logging in to ECR and NGC...
      - REGISTRY=$(echo $ECR_REPO_URI | cut -d/ -f1)
      - aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $REGISTRY
      - NGC_KEY=$(aws secretsmanager get-secret-value --secret-id ngc-api-key --region $AWS_DEFAULT_REGION --query SecretString --output text)
      - echo $NGC_KEY | docker login --username '$oauthtoken' --password-stdin nvcr.io
  build:
    commands:
      - echo Building isaac-lab image...
      - docker build --platform linux/amd64 -t isaac-lab-training .
      - docker tag isaac-lab-training:latest $ECR_REPO_URI
  post_build:
    commands:
      - docker push $ECR_REPO_URI
      - echo Done. Image pushed to $ECR_REPO_URI
"""

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("buildspec.yml", BUILDSPEC)

        # Container files — use the canonical repo paths under containers/isaac-lab/
        # The Dockerfile uses COPY containers/isaac-lab/<file>, so paths must match.
        container_src = REPO_ROOT / "containers" / "isaac-lab"
        for fname in ["sm-train-entrypoint.sh", "batch-train-entrypoint.sh",
                      "train_entrypoint.py"]:
            fpath = container_src / fname
            if fpath.exists():
                arc = f"containers/isaac-lab/{fname}"
                zf.write(fpath, arc)
                log(f"  + {arc} ({fpath.stat().st_size // 1024} KB)")
            else:
                log(f"  ~ {fname} not found (optional)")
        # Dockerfile goes at zip root (docker build context)
        dockerfile = container_src / "Dockerfile"
        if dockerfile.exists():
            zf.write(dockerfile, "Dockerfile")
            log(f"  + Dockerfile ({dockerfile.stat().st_size // 1024} KB)")

        # training/ — all .py and .yaml, excluding data/
        training_dir = REPO_ROOT / "training"
        added = 0
        for p in sorted(training_dir.rglob("*")):
            rel = str(p.relative_to(REPO_ROOT))
            if (p.is_file()
                    and "__pycache__" not in rel
                    and not p.suffix == ".pyc"
                    and "training/data/" not in rel):
                zf.write(p, rel)
                added += 1
        log(f"  + training/ ({added} files, data/ excluded)")

        # workflows/
        for wf in sorted((REPO_ROOT / "workflows").rglob("*")):
            if wf.is_file():
                arc = str(wf.relative_to(REPO_ROOT))
                zf.write(wf, arc)
                log(f"  + {arc}")

    # Verify key files present
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        names = set(zf.namelist())
    must_have = [
        "Dockerfile",
        "containers/isaac-lab/sm-train-entrypoint.sh",
        "containers/isaac-lab/train_entrypoint.py",
        "training/scripts/launch_rl.py",
        "workflows/pick-and-place.yaml",
    ]
    missing = [f for f in must_have if f not in names]
    if missing:
        raise FileNotFoundError(f"Missing from zip: {missing}")
    log(f"  All {len(must_have)} key files present in zip ({len(names)} total)")

    buf.seek(0)
    s3.upload_fileobj(buf, BUCKET, S3_ZIP_KEY)
    log(f"  Uploaded: s3://{BUCKET}/{S3_ZIP_KEY}")


def ensure_codebuild_project():
    """Create or update the CodeBuild project."""
    project_cfg = dict(
        source={"type": "S3", "location": f"{BUCKET}/{S3_ZIP_KEY}", "buildspec": "buildspec.yml"},
        artifacts={"type": "NO_ARTIFACTS"},
        environment={
            "type": "LINUX_CONTAINER",
            "image": "aws/codebuild/standard:7.0",
            "computeType": "BUILD_GENERAL1_2XLARGE",
            "privilegedMode": True,
            "environmentVariables": [
                {"name": "ECR_REPO_URI",       "value": ECR_URI},
                {"name": "AWS_DEFAULT_REGION", "value": REGION},
            ],
        },
        serviceRole=ROLE_ARN,
        timeoutInMinutes=90,
    )
    try:
        cb.create_project(name=CB_PROJECT, **project_cfg)
        log(f"  CodeBuild project created: {CB_PROJECT}")
    except cb.exceptions.ResourceAlreadyExistsException:
        cb.update_project(name=CB_PROJECT, **{k: v for k, v in project_cfg.items() if k != "timeoutInMinutes"})
        log(f"  CodeBuild project updated: {CB_PROJECT}")


def run_build() -> str:
    """Trigger CodeBuild and return build_id."""
    build = cb.start_build(projectName=CB_PROJECT)
    build_id = build["build"]["id"]
    log(f"  Build started: {build_id}")
    log(f"  Monitor: https://{REGION}.console.aws.amazon.com/codesuite/codebuild/projects/{CB_PROJECT}/history")
    return build_id


def wait_build(build_id: str):
    """Poll until build completes. Raise on failure."""
    log(f"Waiting for CodeBuild {build_id} ...")

    def check():
        b = cb.batch_get_builds(ids=[build_id])["builds"][0]
        status = b["buildStatus"]
        elapsed = int((time.time() - b["startTime"].timestamp()) / 60)
        log(f"  Build status: {status} ({elapsed} min elapsed)")
        if status == "SUCCEEDED":
            return b
        if status in ("FAILED", "STOPPED", "TIMED_OUT", "FAULT"):
            # Print failed phase errors
            for ph in b.get("phases", []):
                if ph.get("phaseStatus") == "FAILED":
                    log(f"  FAILED phase: {ph['phaseType']}")
                    for ctx in ph.get("contexts", []):
                        log(f"    {ctx.get('message','')}")
            raise RuntimeError(f"Build {build_id} failed with status: {status}")
        return None

    poll(f"CodeBuild {build_id}", check, interval=60, timeout=5400)
    log("✅ Build SUCCEEDED")


def do_build():
    build_zip()
    ensure_codebuild_project()
    build_id = run_build()
    wait_build(build_id)
    return build_id


# ── Step 2: RL Training ────────────────────────────────────────────────────────

def launch_training(task: str, num_envs: int, max_iterations: int,
                    instance_type: str) -> str:
    """Launch training via the repo's canonical launch_rl.py script."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))

    os.environ["AWS_DEFAULT_REGION"]  = REGION
    os.environ["PROJECT_NAME"]        = "physical-ai"
    os.environ["ENVIRONMENT"]         = "dev"
    os.environ["SAGEMAKER_ROLE_ARN"]  = ROLE_ARN
    os.environ["ISAAC_LAB_IMAGE"]     = ECR_URI
    os.environ["DATASETS_BUCKET"]     = BUCKET

    from training.scripts.launch_rl import launch  # noqa: E402

    log(f"  Launching training via launch_rl.py")
    log(f"  Task: {task}, envs: {num_envs}, iters: {max_iterations}, instance: {instance_type}")

    job_name = launch(
        task=task,
        num_envs=num_envs,
        max_iterations=max_iterations,
        framework="rsl_rl",
        instance_type=instance_type,
        instance_count=1,
        runtime_min=max(30, max_iterations * 2),
        dry_run=False,
    )
    log(f"  CW logs: https://console.aws.amazon.com/cloudwatch/home?region={REGION}"
        f"#logsV2:log-groups/log-group/$252Faws$252Fsagemaker$252FTrainingJobs"
        f"/log-events/{job_name}")
    return job_name


def wait_training(job_name: str) -> str:
    """Poll until training completes. Returns model S3 URI."""
    log(f"Waiting for training job {job_name} ...")

    def check():
        desc = sm.describe_training_job(TrainingJobName=job_name)
        status = desc["TrainingJobStatus"]
        secondary = desc.get("SecondaryStatus", "")
        log(f"  Training status: {status} / {secondary}")
        if status == "Completed":
            return desc
        if status == "Failed":
            reason = desc.get("FailureReason", "")[:300]
            # Fetch last 40 lines of CW logs for diagnosis
            cw_tail = get_cw_log_tail(
                "/aws/sagemaker/TrainingJobs",
                job_name,
                lines=40,
            )
            log(f"\n{'='*60}")
            log(f"TRAINING FAILED: {reason}")
            log(f"\nLast CloudWatch log lines:\n{cw_tail}")
            log(f"{'='*60}\n")
            raise RuntimeError(f"Training job {job_name} failed: {reason}")
        return None

    desc = poll(f"training {job_name}", check, interval=60, timeout=7200)
    model_s3 = desc["ModelArtifacts"]["S3ModelArtifacts"]
    duration  = (desc["TrainingEndTime"] - desc["TrainingStartTime"]).seconds // 60
    log(f"✅ Training complete in {duration} min — output: {model_s3}")
    return model_s3


def verify_checkpoint(model_s3: str) -> bool:
    """Download the tar and confirm at least one checkpoint file exists."""
    log("Verifying checkpoint in model artifact ...")
    bucket = model_s3.split("/")[2]
    key    = "/".join(model_s3.split("/")[3:])
    tmp    = Path("/tmp/pipeline-check.tar.gz")
    s3.download_file(bucket, key, str(tmp))
    result = subprocess.run(
        ["tar", "-tzf", str(tmp)],
        capture_output=True, text=True
    )
    files = result.stdout.strip().splitlines()
    log(f"  Artifact contents ({len(files)} files): {files[:10]}")
    checkpoints = [f for f in files if f.endswith(".pt") or f.endswith(".pth")]
    if not checkpoints:
        log("  ❌ No .pt checkpoint found in artifact — training exited without saving")
        return False
    log(f"  ✅ Checkpoints found: {checkpoints}")
    return True


# ── Step 3: Render video ───────────────────────────────────────────────────────

def launch_video(model_s3: str, task: str, video_length: int = 600) -> str:
    job_name = f"isaac-lab-video-{int(time.time())}"
    log(f"  Launching video render job: {job_name}")
    log(f"  Checkpoint: {model_s3}")
    log(f"  Task: {task}, video_length: {video_length} steps")

    sm.create_training_job(
        TrainingJobName=job_name,
        RoleArn=ROLE_ARN,
        AlgorithmSpecification={"TrainingImage": ECR_URI, "TrainingInputMode": "File"},
        InputDataConfig=[{
            "ChannelName": "model",
            "DataSource": {"S3DataSource": {
                "S3DataType":                "S3Prefix",
                "S3Uri":                     model_s3,
                "S3DataDistributionType":    "FullyReplicated",
            }},
        }],
        OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/isaac-lab/videos/"},
        ResourceConfig={
            "InstanceType":   "ml.g5.xlarge",
            "InstanceCount":  1,
            "VolumeSizeInGB": 100,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 1800},
        HyperParameters={
            "mode":         "play",
            "task":         task,
            "framework":    "rsl_rl",
            "video_length": str(video_length),
        },
    )
    log(f"  CW logs: https://console.aws.amazon.com/cloudwatch/home?region={REGION}"
        f"#logsV2:log-groups/log-group/$252Faws$252Fsagemaker$252FTrainingJobs"
        f"/log-events/{job_name}")
    return job_name


def wait_video(job_name: str) -> str:
    """Poll until video render completes. Returns model S3 URI."""
    log(f"Waiting for video render job {job_name} ...")

    def check():
        desc = sm.describe_training_job(TrainingJobName=job_name)
        status = desc["TrainingJobStatus"]
        secondary = desc.get("SecondaryStatus", "")
        log(f"  Video status: {status} / {secondary}")
        if status == "Completed":
            return desc
        if status == "Failed":
            reason = desc.get("FailureReason", "")[:300]
            cw_tail = get_cw_log_tail(
                "/aws/sagemaker/TrainingJobs",
                job_name,
                lines=40,
            )
            log(f"\n{'='*60}")
            log(f"VIDEO RENDER FAILED: {reason}")
            log(f"\nLast CloudWatch log lines:\n{cw_tail}")
            log(f"{'='*60}\n")
            raise RuntimeError(f"Video job {job_name} failed: {reason}")
        return None

    desc = poll(f"video render {job_name}", check, interval=30, timeout=1800)
    output_s3 = desc["ModelArtifacts"]["S3ModelArtifacts"]
    log(f"✅ Video render complete — output: {output_s3}")
    return output_s3


def download_video(output_s3: str) -> Path:
    """Download and extract video artifact. Returns path to first MP4."""
    log("Downloading video artifact ...")
    bucket = output_s3.split("/")[2]
    key    = "/".join(output_s3.split("/")[3:])

    out_dir = Path.home() / "ur3-video"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    tar_path = out_dir / "video.tar.gz"
    s3.download_file(bucket, key, str(tar_path))
    log(f"  Downloaded: {tar_path.stat().st_size / 1e6:.1f} MB")

    subprocess.run(["tar", "-xzf", str(tar_path), "-C", str(out_dir)], check=True)
    mp4s = sorted(out_dir.rglob("*.mp4"))
    if mp4s:
        log(f"✅ Video saved: {mp4s[0]}")
        log(f"   Open from the JupyterLab file browser in ~/ur3-video/")
        return mp4s[0]
    else:
        log("⚠️  No .mp4 found in artifact — check the video render logs")
        log(f"   Extracted contents: {list(out_dir.rglob('*'))[:20]}")
        return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Lab 4 end-to-end pipeline")
    parser.add_argument("--rebuild",         action="store_true",
                        help="Force CodeBuild container rebuild (default: skip if image exists)")
    parser.add_argument("--skip-training",   action="store_true",
                        help="Skip RL training (requires --checkpoint-s3)")
    parser.add_argument("--skip-video",      action="store_true",
                        help="Stop after training, skip video render")
    parser.add_argument("--checkpoint-s3",   type=str, default=None,
                        help="S3 URI of model.tar.gz to use for video (skips training)")
    parser.add_argument("--max-iterations",  type=int, default=100,
                        help="RL training iterations (default: 100)")
    parser.add_argument("--num-envs",        type=int, default=4096,
                        help="Parallel environments (default: 4096)")
    parser.add_argument("--task",            type=str, default="Isaac-Reach-UR10-v0",
                        help="Isaac Lab task name (default: validated UR10 reach task)")
    parser.add_argument("--instance-type",   type=str, default="ml.g5.12xlarge",
                        help="SageMaker instance type (G-family only)")
    parser.add_argument("--video-length",    type=int, default=600,
                        help="Video steps to render (default: 600 = ~12 sec)")
    args = parser.parse_args()

    state = load_state()
    log(f"Pipeline starting — task: {args.task}")
    log(f"  Rebuild:       {args.rebuild}")
    log(f"  Skip training: {args.skip_training}")
    log(f"  Skip video:    {args.skip_video}")
    log(f"  Max iters:     {args.max_iterations}")
    log(f"  Num envs:      {args.num_envs}")
    log(f"  Instance:      {args.instance_type}")

    # ── Build ────────────────────────────────────────────────────────────────
    if args.rebuild:
        log("\n── STEP 1: Build container ──────────────────────────────────────")
        build_id = do_build()
        state["last_build_id"] = build_id
        save_state(state)
    else:
        # Verify ECR image exists
        try:
            imgs = ecr.describe_images(
                repositoryName=ECR_REPO,
                imageIds=[{"imageTag": "latest"}],
            )["imageDetails"]
            pushed = imgs[0]["imagePushedAt"].strftime("%Y-%m-%d %H:%M UTC")
            size_gb = imgs[0]["imageSizeInBytes"] / 1e9
            log(f"\n── STEP 1: Skipping build (--rebuild not set)")
            log(f"   Using existing ECR image: {size_gb:.1f} GB, pushed {pushed}")
        except Exception:
            log("  ❌ No ECR image found — run with --rebuild first")
            sys.exit(1)

    # ── Training ─────────────────────────────────────────────────────────────
    if args.skip_training:
        if args.checkpoint_s3:
            model_s3 = args.checkpoint_s3
            log(f"\n── STEP 2: Skipping training — using checkpoint: {model_s3}")
        else:
            log("  ❌ --skip-training requires --checkpoint-s3")
            sys.exit(1)
    else:
        log("\n── STEP 2: RL Training ──────────────────────────────────────────")
        job_name = launch_training(
            task=args.task,
            num_envs=args.num_envs,
            max_iterations=args.max_iterations,
            instance_type=args.instance_type,
        )
        state["last_training_job"] = job_name
        save_state(state)

        model_s3 = wait_training(job_name)
        state["last_model_s3"] = model_s3
        save_state(state)

        if not verify_checkpoint(model_s3):
            log("  ❌ Training completed but no checkpoint was saved.")
            log("     Check the CloudWatch logs above for the actual error.")
            log("     Fix the issue and re-run (no need to --rebuild).")
            sys.exit(1)

    # ── Video ─────────────────────────────────────────────────────────────────
    if args.skip_video:
        log(f"\n── STEP 3: Skipping video render (--skip-video set)")
        log(f"   Checkpoint at: {model_s3}")
        log(f"   Re-run with --skip-training --checkpoint-s3 '{model_s3}' to render later")
    else:
        log("\n── STEP 3: Render video ─────────────────────────────────────────")
        video_job = launch_video(
            model_s3=model_s3,
            task=args.task,
            video_length=args.video_length,
        )
        state["last_video_job"] = video_job
        save_state(state)

        video_output_s3 = wait_video(video_job)
        state["last_video_s3"] = video_output_s3
        save_state(state)

        mp4 = download_video(video_output_s3)
        if mp4:
            state["last_mp4"] = str(mp4)
            save_state(state)

    log("\n── Pipeline complete ────────────────────────────────────────────")
    log(f"   State saved to: {STATE_FILE}")
    log(f"   Re-run with --skip-training --checkpoint-s3 '{state.get('last_model_s3','')}' to re-render")


if __name__ == "__main__":
    main()
