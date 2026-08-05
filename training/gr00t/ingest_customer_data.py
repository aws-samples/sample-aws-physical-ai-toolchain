"""
Bring-Your-Own-Data ingestion for GR00T fine-tuning.

One command to take a customer's raw Zarr teleop recordings all the way to a
training-ready dataset in S3 (and, optionally, kick off training):

    raw Zarr episodes  →  convert to LeRobot v2  →  upload to S3  →  [launch training]

This is the on-ramp for customers who want to train on THEIR OWN demonstrations
rather than the bundled UR3 demo. The expected Zarr schema is documented in
docs/zarr-schema.md.

Usage:
    # Convert + upload your episodes (bucket auto-detected from the Foundation stack):
    python training/gr00t/ingest_customer_data.py \
        --episodes-dir ./my_robot_episodes \
        --prefix groot-data/myrobot

    # ...and immediately launch a 100-step smoke training run:
    python training/gr00t/ingest_customer_data.py \
        --episodes-dir ./my_robot_episodes \
        --prefix groot-data/myrobot \
        --train --max-steps 100

    # See exactly what would happen without touching AWS or converting:
    python training/gr00t/ingest_customer_data.py \
        --episodes-dir ./my_robot_episodes --prefix groot-data/myrobot --dry-run

WORKSHOP NOTE: conversion needs the data deps (zarr, opencv, pandas, pyarrow):
    pip install -r training/requirements.txt
"""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONVERT_SCRIPT = HERE / "convert_zarr_to_lerobot.py"
LAUNCH_SCRIPT = HERE / "launch_training.py"
STACK_NAME = "PhysicalAi-dev-Foundation"


def get_stack_output(key: str, stack_name: str = STACK_NAME) -> str:
    """Read a single CloudFormation stack output (via the AWS CLI).

    Security: key and stack_name are hardcoded constants or argparse inputs.
    List-form subprocess (no shell=True) prevents injection.
    """
    cmd = ["aws", "cloudformation", "describe-stacks", "--stack-name", stack_name,
           "--query", f"Stacks[0].Outputs[?OutputKey=='{key}'].OutputValue", "--output", "text"]
    result = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603 # nosemgrep: dangerous-subprocess-use-audit
    if result.returncode != 0:
        raise RuntimeError(f"Could not read stack output {key}: {result.stderr.strip()}")
    val = result.stdout.strip()
    if not val or val == "None":
        raise RuntimeError(f"Stack output {key} not found in {stack_name}")
    return val


def plan(episodes_dir: str, output_dir: str, prefix: str, train: bool, max_steps: int) -> dict:
    """Describe the steps without running them. PURE — no AWS, no conversion."""
    steps = [
        {"step": "convert", "cmd": [
            sys.executable, str(CONVERT_SCRIPT),
            "--episodes-dir", episodes_dir, "--output-dir", output_dir]},
        {"step": "upload", "cmd": [
            "aws", "s3", "sync", output_dir, f"s3://<BUCKET>/{prefix}/dataset/"]},
    ]
    if train:
        steps.append({"step": "train", "cmd": [
            sys.executable, str(LAUNCH_SCRIPT),
            "--s3-bucket", "<BUCKET>", "--dataset-prefix", prefix,
            "--role-arn", "<ROLE_ARN>", "--ecr-image", "<ECR_URI>:latest",
            "--max-steps", str(max_steps)]})
    return {"dry_run": True, "prefix": prefix, "steps": steps}


def run_convert(episodes_dir: str, output_dir: str) -> None:
    print(f"\n  [1] Converting Zarr → LeRobot v2: {episodes_dir} → {output_dir}")
    cmd = [sys.executable, str(CONVERT_SCRIPT),
           "--episodes-dir", episodes_dir, "--output-dir", output_dir]
    subprocess.run(cmd, check=True)  # noqa: S603 # nosemgrep: dangerous-subprocess-use-audit


def run_upload(output_dir: str, bucket: str, prefix: str) -> str:
    s3_uri = f"s3://{bucket}/{prefix}/dataset/"
    print(f"\n  [2] Uploading to {s3_uri}")
    cmd = ["aws", "s3", "sync", output_dir, s3_uri, "--quiet"]
    subprocess.run(cmd, check=True)  # noqa: S603 # nosemgrep: dangerous-subprocess-use-audit
    return s3_uri


def run_train(bucket: str, prefix: str, role_arn: str, ecr_image: str, max_steps: int, region: str) -> None:
    print(f"\n  [3] Launching training ({max_steps} steps)")
    cmd = [sys.executable, str(LAUNCH_SCRIPT),
           "--s3-bucket", bucket, "--dataset-prefix", prefix,
           "--role-arn", role_arn, "--ecr-image", f"{ecr_image}:latest",
           "--max-steps", str(max_steps), "--region", region]
    subprocess.run(cmd, check=True)  # noqa: S603 # nosemgrep: dangerous-subprocess-use-audit


def main():
    p = argparse.ArgumentParser(description="Ingest customer Zarr data for GR00T fine-tuning")
    p.add_argument("--episodes-dir", required=True, help="Dir of raw Zarr episode_* folders")
    p.add_argument("--output-dir", default=None, help="LeRobot output dir (default: <episodes-dir>_lerobot)")
    p.add_argument("--prefix", default="groot-data/custom", help="S3 prefix under the datasets bucket")
    p.add_argument("--bucket", default=None, help="Datasets bucket (auto-detected from CDK if omitted)")
    p.add_argument("--stack-name", default=STACK_NAME)
    p.add_argument("--train", action="store_true", help="Launch training after upload")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--region", default="us-west-2")
    p.add_argument("--dry-run", action="store_true", help="Show the plan; make no AWS calls and convert nothing")
    args = p.parse_args()

    output_dir = args.output_dir or f"{args.episodes_dir.rstrip('/')}_lerobot"

    if args.dry_run:
        result = plan(args.episodes_dir, output_dir, args.prefix, args.train, args.max_steps)
        print(json.dumps(result, indent=2))
        print("\n  [dry-run] No AWS calls made. No data converted.")
        return

    # Validate input early.
    if not Path(args.episodes_dir).is_dir():
        print(f"  ERROR: episodes dir not found: {args.episodes_dir}", file=sys.stderr)
        sys.exit(1)

    bucket = args.bucket or get_stack_output("DatasetsBucketName", args.stack_name)

    run_convert(args.episodes_dir, output_dir)
    s3_uri = run_upload(output_dir, bucket, args.prefix)

    result = {"status": "ingested", "s3_uri": s3_uri, "prefix": args.prefix}

    if args.train:
        role_arn = get_stack_output("SageMakerRoleArn", args.stack_name)
        ecr_image = get_stack_output("GrootTrainingRepoUri", args.stack_name)
        run_train(bucket, args.prefix, role_arn, ecr_image, args.max_steps, args.region)
        result["status"] = "ingested_and_training"

    print("\n" + json.dumps(result, indent=2))
    if not args.train:
        print(f"\n  Next: launch training with\n"
              f"    python training/gr00t/pipeline.py --execute "
              f"--dataset-prefix {args.prefix} --max-steps 100 --region {args.region}")


if __name__ == "__main__":
    main()
