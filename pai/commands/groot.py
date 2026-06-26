"""pai groot — Launch GR00T fine-tuning jobs on SageMaker."""

import os
from pathlib import Path

import click

from pai import cfn, config, helpers


@click.group()
def groot():
    """Launch GR00T fine-tuning jobs."""
    pass


@groot.command()
@click.option("--dataset-dir", default="training/data/ur3_lerobot_dataset", help="Path to LeRobot dataset directory")
@click.option("--max-steps", type=int, default=5000, help="Maximum training steps")
@click.option("--dry-run", is_flag=True, help="Preview the job without making AWS calls")
def launch(dataset_dir, max_steps, dry_run):
    """Launch a GR00T fine-tuning job on SageMaker (Path A).

    Ports the logic from run-path-a.sh:
    1. Read Foundation stack outputs (bucket, role, ECR)
    2. Upload dataset to S3
    3. Launch SageMaker GR00T training job
    """
    from datetime import datetime

    import boto3

    region = config.resolve_region()
    stack_name = "PhysicalAi-dev-Foundation"
    prefix = "groot-data/ur3"  # matches Lab 1 + pipeline.py defaults

    helpers.heading("AWS Physical AI Toolchain — Path A")
    helpers.heading("GR00T Fine-Tuning on SageMaker")

    # Check for HF_TOKEN
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        helpers.warn("\n⚠️  WARNING: HF_TOKEN not set in environment.")
        helpers.warn("The GR00T base model download may fail or hit rate limits.")
        helpers.warn("Set HF_TOKEN before launching: export HF_TOKEN=<your-token>\n")

    # Step 1: Read CDK Outputs
    helpers.info("[1/4] Reading infrastructure outputs...")
    try:
        bucket = cfn.bucket(stack_name, region)
        role_arn = cfn.role_arn(stack_name, region)
        # Get GR00T training ECR URI (Foundation stack output key)
        outputs = cfn.stack_outputs(stack_name, region)
        ecr_uri = outputs.get("GrootTrainingRepoUri", "")

        if not bucket or not role_arn or not ecr_uri:
            helpers.error(f"Failed to read Foundation stack outputs. Is the stack deployed?")
            helpers.info(f"  Bucket:   {bucket or 'MISSING'}")
            helpers.info(f"  Role:     {role_arn or 'MISSING'}")
            helpers.info(f"  ECR:      {ecr_uri or 'MISSING'}")
            return

        helpers.info(f"  Bucket:   {bucket}")
        helpers.info(f"  Role:     {role_arn}")
        helpers.info(f"  ECR:      {ecr_uri}")
    except Exception as e:
        helpers.friendly_boto_error(e)
        return

    # Step 2: Check dataset exists
    helpers.info("\n[2/4] Checking dataset...")
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        helpers.error(f"Dataset not found at: {dataset_dir}")
        helpers.info(f"Download with: python training/groot/download_demo_dataset.py --output {dataset_dir}")
        return
    helpers.info(f"  Found: {dataset_dir}")

    # Step 3: Upload to S3 (skip if dry-run)
    s3_uri = f"s3://{bucket}/{prefix}/dataset/"
    if dry_run:
        helpers.info(f"\n[3/4] [dry-run] Would upload dataset to: {s3_uri}")
    else:
        helpers.info(f"\n[3/4] Uploading dataset to S3...")
        try:
            s3 = boto3.client("s3", region_name=region)
            # Upload all files in dataset directory
            for file_path in dataset_path.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(dataset_path)
                    s3_key = f"{prefix}/dataset/{rel_path}"
                    s3.upload_file(str(file_path), bucket, s3_key)
            helpers.info(f"  Uploaded to: {s3_uri}")
        except Exception as e:
            helpers.friendly_boto_error(e)
            return

    # Step 4: Launch SageMaker job
    helpers.info("\n[4/4] Launching SageMaker training job...")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"groot-finetune-{timestamp}"
    image = f"{ecr_uri}:latest"

    training_config = {
        "TrainingJobName": job_name,
        "RoleArn": role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": image,
            "TrainingInputMode": "File",
        },
        "HyperParameters": {
            "base_model": "nvidia/GR00T-N1.6-3B",
            "max_steps": str(max_steps),
            "batch_size": "8",
            "learning_rate": "1e-4",
            "dataset_path": "/opt/ml/input/data/training/dataset",
        },
        "InputDataConfig": [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": s3_uri,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        ],
        "OutputDataConfig": {
            "S3OutputPath": f"s3://{bucket}/{prefix}/output/",
        },
        "ResourceConfig": {
            "InstanceType": "ml.g5.12xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 200,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 86400,  # 24 hours max
        },
    }

    # Add HF_TOKEN if available
    if hf_token:
        training_config["Environment"] = {
            "HF_TOKEN": hf_token,
            "HUGGING_FACE_HUB_TOKEN": hf_token,
        }

    if dry_run:
        import copy
        import json
        # Redact secrets before printing — never echo the HF token to stdout/logs.
        printable = copy.deepcopy(training_config)
        if "Environment" in printable:
            printable["Environment"] = {
                k: "<redacted>" for k in printable["Environment"]
            }
        helpers.info("\n[dry-run] Would call sagemaker.create_training_job with:\n")
        click.echo(json.dumps(printable, indent=2, default=str))
        helpers.info("\n[dry-run] No AWS calls made.")
        helpers.info(f"\nResolved configuration:")
        helpers.info(f"  Job name:     {job_name}")
        helpers.info(f"  Image:        {image}")
        helpers.info(f"  Dataset:      {s3_uri}")
        helpers.info(f"  Output:       s3://{bucket}/{prefix}/output/")
        helpers.info(f"  Max steps:    {max_steps}")
    else:
        try:
            sm = boto3.client("sagemaker", region_name=region)
            sm.create_training_job(**training_config)
            helpers.success(f"\n{'='*60}")
            helpers.success(f"Launched SageMaker job: {job_name}")
            helpers.info(f"\nMonitor with:")
            helpers.info(f"  pai rl status {job_name}")
            helpers.info(f"  aws sagemaker describe-training-job --training-job-name {job_name} --region {region}")
            helpers.success(f"{'='*60}")
        except Exception as e:
            helpers.friendly_boto_error(e)


def register(cli: click.Group):
    """Register groot commands with the CLI."""
    cli.add_command(groot)
