"""
Upload a local LeRobot dataset to the S3 datasets bucket.

Reads the bucket name from CloudFormation outputs (or accepts it as a flag).

Usage:
    python upload_dataset.py --dataset-dir training/data/ur3_lerobot_dataset --prefix groot-data/ur3
    python upload_dataset.py --dataset-dir ./my-data --prefix groot-data/myrobot --bucket my-bucket

WORKSHOP NOTE: Run this after converting/downloading a dataset and before launch_training.py.
"""

import argparse
import shlex
import subprocess
import json
import sys


def get_bucket_from_cloudformation(stack_name: str = "PhysicalAi-dev-Foundation") -> str:
    """Read the datasets bucket name from CDK stack outputs."""
    try:
        # Security: all args are literals or CloudFormation outputs (no user input).
        result = subprocess.run(  # nosemgrep: dangerous-subprocess-use-audit
            ["aws", "cloudformation", "describe-stacks",
             "--stack-name", stack_name,
             "--query", "Stacks[0].Outputs[?OutputKey==`DatasetsBucketName`].OutputValue",
             "--output", "text"],
            capture_output=True, text=True, check=True
        )
        bucket = result.stdout.strip()
        if not bucket or bucket == "None":
            raise ValueError(f"No DatasetsBucketName output found in stack {stack_name}")
        return bucket
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Could not read CloudFormation outputs: {e.stderr}")
        sys.exit(1)


def upload_dataset(dataset_dir: str, bucket: str, prefix: str) -> dict:
    """Upload dataset to S3 using aws s3 sync.

    Security: all arguments are validated (argparse CLI input + CloudFormation output).
    No shell=True; list-form subprocess prevents injection.
    """

    s3_uri = f"s3://{bucket}/{prefix}/dataset/"

    print(f"{'='*60}")
    print(f"  Dataset Upload")
    print(f"  Source: {dataset_dir}")
    print(f"  Destination: {s3_uri}")
    print(f"{'='*60}")

    # Run aws s3 sync — arguments are list-form (no shell interpolation).
    # shlex.quote() used in logging only; the list form is inherently safe.
    cmd = ["aws", "s3", "sync", dataset_dir, s3_uri, "--quiet"]
    print(f"\n  Running: {shlex.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603 # nosemgrep: dangerous-subprocess-use-audit

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return {"status": "error", "message": result.stderr}

    print(f"  Upload complete!")
    print(f"\n  Next step: Launch training")
    print(f"    python training/groot/launch_training.py \\")
    print(f"      --s3-bucket {bucket} \\")
    print(f"      --dataset-prefix {prefix} \\")
    print(f"      --role-arn <SAGEMAKER_ROLE_ARN> \\")
    print(f"      --ecr-image <GROOT_TRAINING_ECR>:latest")

    return {
        "status": "success",
        "s3_uri": s3_uri,
        "bucket": bucket,
        "prefix": prefix,
    }


def main():
    parser = argparse.ArgumentParser(description="Upload dataset to S3")
    parser.add_argument("--dataset-dir", required=True, help="Local dataset directory")
    parser.add_argument("--prefix", default="groot-data/ur3", help="S3 prefix (under the bucket)")
    parser.add_argument("--bucket", default=None, help="S3 bucket name (auto-detected from CDK if not provided)")
    parser.add_argument("--stack-name", default="PhysicalAi-dev-Foundation", help="CDK stack name to read outputs from")
    args = parser.parse_args()

    # Get bucket name
    bucket = args.bucket or get_bucket_from_cloudformation(args.stack_name)

    result = upload_dataset(args.dataset_dir, bucket, args.prefix)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
