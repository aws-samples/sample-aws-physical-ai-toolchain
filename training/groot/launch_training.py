"""
Launch GR00T Fine-Tuning on SageMaker (Path A)

Submits a SageMaker Training Job that fine-tunes NVIDIA GR00T N1.6-3B
on a customer's LeRobot-format dataset stored in S3.

Usage:
    python launch_training.py \
        --s3-bucket physical-ai-dev-datasets-123456789 \
        --dataset-prefix groot-data/lab1 \
        --role-arn arn:aws:iam::123456789:role/physical-ai-dev-sagemaker-role \
        --ecr-image 123456789.dkr.ecr.us-west-2.amazonaws.com/physical-ai/groot-training:latest \
        --max-steps 5000

    # Dry run (shows config, doesn't launch):
    python launch_training.py --dry-run ...
"""

import argparse
import json
import time
from datetime import datetime

import boto3


def launch_training_job(
    s3_bucket: str,
    dataset_prefix: str,
    role_arn: str,
    ecr_image: str,
    max_steps: int = 5000,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    base_model: str = "nvidia/GR00T-N1.6-3B",
    instance_type: str = "ml.g5.12xlarge",
    region: str = "us-west-2",
    dry_run: bool = False,
) -> dict:
    """Submit a SageMaker training job for GR00T fine-tuning."""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"groot-finetune-{timestamp}"

    training_config = {
        "TrainingJobName": job_name,
        "RoleArn": role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": ecr_image,
            "TrainingInputMode": "File",
        },
        "HyperParameters": {
            "base_model": base_model,
            "max_steps": str(max_steps),
            "batch_size": str(batch_size),
            "learning_rate": str(learning_rate),
            "dataset_path": "/opt/ml/input/data/training/dataset",
        },
        "InputDataConfig": [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{s3_bucket}/{dataset_prefix}/dataset/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        ],
        "OutputDataConfig": {
            "S3OutputPath": f"s3://{s3_bucket}/{dataset_prefix}/output/",
        },
        "ResourceConfig": {
            "InstanceType": instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 200,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 86400,  # 24 hours max
        },
    }

    # Inject the HuggingFace token from the environment (both names the SDK reads).
    # Never bake a token or a placeholder into the request.
    import os
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        training_config["Environment"] = {
            "HF_TOKEN": hf_token,
            "HUGGING_FACE_HUB_TOKEN": hf_token,
        }
    else:
        print("  WARNING: HF_TOKEN not set — the base-model download may hit rate limits.")

    # Cost estimation
    # ml.g5.12xlarge: ~$7.09/hr (on-demand, us-west-2, June 2026)
    estimated_hours = (max_steps * 8) / 3600  # ~8 sec/step
    estimated_cost = estimated_hours * 7.09

    if dry_run:
        return {
            "dry_run": True,
            "job_name": job_name,
            "config": training_config,
            "estimated_duration_hours": round(estimated_hours, 1),
            "estimated_cost_usd": round(estimated_cost, 0),
            "note": "Drop --dry-run to launch.",
        }

    # Launch the job
    sm_client = boto3.client("sagemaker", region_name=region)
    sm_client.create_training_job(**training_config)

    # Wait for job to start
    print(f"  Job submitted: {job_name}")
    print(f"  Estimated duration: ~{estimated_hours:.1f} hours")
    print(f"  Estimated cost: ~${estimated_cost:.0f}")
    print(f"  Monitor: aws sagemaker describe-training-job --training-job-name {job_name}")

    return {
        "status": "launched",
        "job_name": job_name,
        "estimated_duration_hours": round(estimated_hours, 1),
        "estimated_cost_usd": round(estimated_cost, 0),
        "model_output": f"s3://{s3_bucket}/{dataset_prefix}/output/{job_name}/output/model.tar.gz",
    }


def main():
    parser = argparse.ArgumentParser(description="Launch GR00T fine-tuning on SageMaker")
    parser.add_argument("--s3-bucket", required=True, help="S3 bucket with training data")
    parser.add_argument("--dataset-prefix", required=True, help="S3 prefix for the dataset")
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--ecr-image", required=True, help="ECR URI for training container")
    parser.add_argument("--max-steps", type=int, default=5000, help="Training steps (5000=~11hrs)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--base-model", default="nvidia/GR00T-N1.6-3B")
    parser.add_argument("--instance-type", default="ml.g5.12xlarge")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--dry-run", action="store_true", help="Show config without launching")
    args = parser.parse_args()

    result = launch_training_job(
        s3_bucket=args.s3_bucket,
        dataset_prefix=args.dataset_prefix,
        role_arn=args.role_arn,
        ecr_image=args.ecr_image,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        base_model=args.base_model,
        instance_type=args.instance_type,
        region=args.region,
        dry_run=args.dry_run,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
