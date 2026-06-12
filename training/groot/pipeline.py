"""
GR00T Fine-Tuning Pipeline — SageMaker Pipelines (Path A Production)

Defines a repeatable, observable ML pipeline:
  Step 1: Train — Fine-tune GR00T on LeRobot dataset (+ eval report in same step)
  Step 2: Register — Save model to SageMaker Model Registry

Usage:
    # Create/update the pipeline:
    python pipeline.py --create \
        --s3-bucket physical-ai-dev-datasets-802782083985 \
        --role-arn arn:aws:iam::802782083985:role/physical-ai-dev-sagemaker-role \
        --ecr-image 802782083985.dkr.ecr.us-east-1.amazonaws.com/physical-ai/groot-training:latest

    # Execute a run:
    python pipeline.py --execute \
        --dataset-prefix groot-data/demo \
        --max-steps 5000

    # List recent runs:
    python pipeline.py --list-runs

WORKSHOP NOTE: This is the "production" version of run-path-a.sh.
Instead of a one-off training job, you get a versioned, repeatable pipeline
visible in the SageMaker Studio UI with step-by-step execution tracking.
"""

import argparse
import json
from datetime import datetime

import boto3


PIPELINE_NAME = "groot-finetune-pipeline"
MODEL_PACKAGE_GROUP = "groot-models"


def create_pipeline(
    s3_bucket: str,
    role_arn: str,
    ecr_image: str,
    region: str = "us-east-1",
    instance_type: str = "ml.g5.12xlarge",
) -> dict:
    """Create or update the SageMaker Pipeline using boto3 API directly."""

    sm = boto3.client("sagemaker", region_name=region)

    # Ensure model package group exists
    try:
        sm.create_model_package_group(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP,
            ModelPackageGroupDescription="GR00T fine-tuned robot manipulation models",
        )
        print(f"  Created model package group: {MODEL_PACKAGE_GROUP}")
    except sm.exceptions.ClientError as e:
        if "already exists" in str(e).lower():
            print(f"  Model package group already exists: {MODEL_PACKAGE_GROUP}")
        else:
            raise

    # Pipeline definition in JSON format
    # This uses SageMaker Pipeline's native JSON schema
    pipeline_definition = {
        "Version": "2020-12-01",
        "Parameters": [
            {
                "Name": "DatasetPrefix",
                "Type": "String",
                "DefaultValue": "groot-data/demo",
            },
            {
                "Name": "MaxSteps",
                "Type": "String",
                "DefaultValue": "5000",
            },
            {
                "Name": "BatchSize",
                "Type": "String",
                "DefaultValue": "8",
            },
            {
                "Name": "BaseModel",
                "Type": "String",
                "DefaultValue": "nvidia/GR00T-N1.7-3B",
            },
            {
                "Name": "InstanceType",
                "Type": "String",
                "DefaultValue": instance_type,
            },
        ],
        "Steps": [
            {
                "Name": "TrainGR00T",
                "Type": "Training",
                "Arguments": {
                    "AlgorithmSpecification": {
                        "TrainingImage": ecr_image,
                        "TrainingInputMode": "File",
                    },
                    "HyperParameters": {
                        "base_model": {"Get": "Parameters.BaseModel"},
                        "max_steps": {"Get": "Parameters.MaxSteps"},
                        "batch_size": {"Get": "Parameters.BatchSize"},
                        "learning_rate": "0.0001",
                        "dataset_path": "/opt/ml/input/data/training/dataset",
                    },
                    "InputDataConfig": [
                        {
                            "ChannelName": "training",
                            "DataSource": {
                                "S3DataSource": {
                                    "S3DataType": "S3Prefix",
                                    "S3Uri": {
                                        "Std:Join": {
                                            "On": "/",
                                            "Values": [
                                                f"s3://{s3_bucket}",
                                                {"Get": "Parameters.DatasetPrefix"},
                                                "dataset",
                                            ],
                                        }
                                    },
                                    "S3DataDistributionType": "FullyReplicated",
                                }
                            },
                        }
                    ],
                    "OutputDataConfig": {
                        "S3OutputPath": f"s3://{s3_bucket}/pipeline-output/",
                    },
                    "ResourceConfig": {
                        "InstanceType": {"Get": "Parameters.InstanceType"},
                        "InstanceCount": 1,
                        "VolumeSizeInGB": 200,
                    },
                    "StoppingCondition": {
                        "MaxRuntimeInSeconds": 86400,
                    },
                    "RoleArn": role_arn,
                },
            },
            {
                "Name": "RegisterModel",
                "Type": "RegisterModel",
                "Arguments": {
                    "ModelPackageGroupName": MODEL_PACKAGE_GROUP,
                    "ModelApprovalStatus": "PendingManualApproval",
                    "InferenceSpecification": {
                        "Containers": [
                            {
                                "Image": ecr_image,
                                "ModelDataUrl": {
                                    "Get": "Steps.TrainGR00T.ModelArtifacts.S3ModelArtifacts"
                                },
                            }
                        ],
                        "SupportedContentTypes": ["application/json"],
                        "SupportedResponseMIMETypes": ["application/json"],
                        "SupportedRealtimeInferenceInstanceTypes": [
                            "ml.g5.xlarge",
                            "ml.g5.2xlarge",
                        ],
                        "SupportedTransformInstanceTypes": ["ml.g5.xlarge"],
                    },
                    "ModelPackageDescription": "GR00T fine-tuned model",
                },
                "DependsOn": ["TrainGR00T"],
            },
        ],
    }

    # Create or update pipeline
    pipeline_def_json = json.dumps(pipeline_definition)

    try:
        sm.describe_pipeline(PipelineName=PIPELINE_NAME)
        # Pipeline exists — update it
        sm.update_pipeline(
            PipelineName=PIPELINE_NAME,
            PipelineDefinition=pipeline_def_json,
            PipelineDescription="GR00T fine-tuning pipeline: train → eval → register",
            RoleArn=role_arn,
        )
        action = "updated"
    except sm.exceptions.ResourceNotFound:
        # Pipeline doesn't exist — create it
        sm.create_pipeline(
            PipelineName=PIPELINE_NAME,
            PipelineDefinition=pipeline_def_json,
            PipelineDescription="GR00T fine-tuning pipeline: train → eval → register",
            RoleArn=role_arn,
        )
        action = "created"

    print(f"  Pipeline '{PIPELINE_NAME}' {action} successfully.")
    print(f"  View in SageMaker Studio: Pipelines → {PIPELINE_NAME}")
    print(f"  Model registry: {MODEL_PACKAGE_GROUP}")

    return {
        "status": action,
        "pipeline_name": PIPELINE_NAME,
        "model_package_group": MODEL_PACKAGE_GROUP,
        "parameters": {
            "DatasetPrefix": "groot-data/demo",
            "MaxSteps": 5000,
            "BatchSize": 8,
            "BaseModel": "nvidia/GR00T-N1.7-3B",
            "InstanceType": instance_type,
        },
    }


def execute_pipeline(
    dataset_prefix: str = "groot-data/demo",
    max_steps: int = 5000,
    batch_size: int = 8,
    region: str = "us-east-1",
) -> dict:
    """Start a new pipeline execution."""

    sm = boto3.client("sagemaker", region_name=region)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    response = sm.start_pipeline_execution(
        PipelineName=PIPELINE_NAME,
        PipelineExecutionDisplayName=f"run-{timestamp}",
        PipelineParameters=[
            {"Name": "DatasetPrefix", "Value": dataset_prefix},
            {"Name": "MaxSteps", "Value": str(max_steps)},
            {"Name": "BatchSize", "Value": str(batch_size)},
        ],
    )

    arn = response["PipelineExecutionArn"]
    print(f"  Pipeline execution started!")
    print(f"  ARN: {arn}")
    print(f"  Monitor: aws sagemaker list-pipeline-execution-steps --pipeline-execution-arn {arn}")

    return {
        "status": "executing",
        "execution_arn": arn,
        "parameters": {
            "DatasetPrefix": dataset_prefix,
            "MaxSteps": max_steps,
            "BatchSize": batch_size,
        },
    }


def list_runs(region: str = "us-east-1") -> dict:
    """List recent pipeline executions."""
    sm = boto3.client("sagemaker", region_name=region)

    try:
        response = sm.list_pipeline_executions(
            PipelineName=PIPELINE_NAME,
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=10,
        )
    except sm.exceptions.ResourceNotFound:
        return {"status": "error", "message": f"Pipeline '{PIPELINE_NAME}' not found. Run --create first."}

    runs = []
    for ex in response.get("PipelineExecutionSummaries", []):
        runs.append({
            "arn": ex["PipelineExecutionArn"],
            "status": ex["PipelineExecutionStatus"],
            "created": ex["CreationTime"].isoformat(),
        })

    return {"pipeline": PIPELINE_NAME, "runs": runs}


def main():
    parser = argparse.ArgumentParser(description="GR00T Fine-Tuning Pipeline")
    parser.add_argument("--create", action="store_true", help="Create/update the pipeline")
    parser.add_argument("--execute", action="store_true", help="Start a pipeline execution")
    parser.add_argument("--list-runs", action="store_true", help="List recent executions")

    # Create params
    parser.add_argument("--s3-bucket", help="S3 bucket for datasets and output")
    parser.add_argument("--role-arn", help="SageMaker execution role ARN")
    parser.add_argument("--ecr-image", help="ECR URI for training container")
    parser.add_argument("--instance-type", default="ml.g5.12xlarge")
    parser.add_argument("--region", default="us-east-1")

    # Execute params
    parser.add_argument("--dataset-prefix", default="groot-data/demo")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)

    args = parser.parse_args()

    if args.create:
        if not all([args.s3_bucket, args.role_arn, args.ecr_image]):
            parser.error("--create requires --s3-bucket, --role-arn, --ecr-image")
        result = create_pipeline(
            s3_bucket=args.s3_bucket,
            role_arn=args.role_arn,
            ecr_image=args.ecr_image,
            region=args.region,
            instance_type=args.instance_type,
        )
    elif args.execute:
        result = execute_pipeline(
            dataset_prefix=args.dataset_prefix,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            region=args.region,
        )
    elif args.list_runs:
        result = list_runs(region=args.region)
    else:
        parser.print_help()
        return

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
