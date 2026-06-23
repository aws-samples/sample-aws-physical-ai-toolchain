"""
Isaac Lab RL Training Launcher

This script launches Isaac Lab RL training jobs on SageMaker for the UR3 pick-and-place task.

The script provides a simple interface to launch RL training:
- Imitation learning (GR00T/VLA, Lab 1) and RL (Isaac Lab, this lab) are separate
  pipelines for obtaining robot policies
- This script launches the RL pipeline (PPO in Isaac Lab with domain randomization)
- VLA training happens on SageMaker; Isaac Sim + RL jobs run on GPU EC2 instances

Usage:
    # Launch RL training (primary use case)
    python groot_to_rl_bridge.py rl-refine

    # Check status of Lab 1 model (optional, informational only)
    python groot_to_rl_bridge.py status

    # Legacy end-to-end command (now runs RL directly without GR00T dependency)
    python groot_to_rl_bridge.py end-to-end

Optional experimental paths (not standard Physical AI patterns):
    # Pre-train MLP via behavioral cloning on teleop data
    python groot_to_rl_bridge.py pretrain-mlp

Note: Loading a GR00T checkpoint (3B diffusion transformer) into an RL MLP
via load_state_dict(strict=False) loads zero matching tensors due to incompatible
architectures. Warm-starting RL from a prior RL checkpoint of the same architecture
works via the --pretrained flag in train.py.
"""

import argparse
import json
import os
import sys
import boto3
import numpy as np
from pathlib import Path

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
PROJECT_NAME = os.environ.get("PROJECT_NAME", "physical-ai")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "dev")

# Account/resource names are resolved from the caller's identity so this works in
# any account — no hardcoded account ID. Override with env vars if you customized
# the CDK projectName/environment.
ACCOUNT_ID = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
BUCKET = os.environ.get("DATASETS_BUCKET", f"{PROJECT_NAME}-{ENVIRONMENT}-datasets-{ACCOUNT_ID}")
ROLE_ARN = os.environ.get("SAGEMAKER_ROLE_ARN", f"arn:aws:iam::{ACCOUNT_ID}:role/{PROJECT_NAME}-{ENVIRONMENT}-sagemaker-role")
ISAAC_LAB_IMAGE = os.environ.get("ISAAC_LAB_IMAGE", f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{PROJECT_NAME}/isaac-lab:latest")


def get_latest_model_package():
    """Get the latest GR00T model from Model Registry."""
    sm = boto3.client("sagemaker", region_name=REGION)
    
    packages = sm.list_model_packages(
        ModelPackageGroupName="groot-models",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    
    if not packages["ModelPackageSummaryList"]:
        print("  ERROR: No models found in 'groot-models' SageMaker Model Registry")
        print("  The 'status' action inspects the GR00T (Lab 1) model registry.")
        print("  Note: the RL pipeline (rl-refine / end-to-end) does NOT require a GR00T")
        print("  model — RL in Isaac Lab is a separate pipeline. This check is informational.")
        print("")
        print("  ACTION REQUIRED (only if you want a registered GR00T model): complete Lab 1:")
        print("    cd /home/ec2-user/projects/physical-ai-toolchain")
        print("    python training/scripts/register_model.py --model-data s3://.../output/model.tar.gz")
        print("")
        print("  Or run the full Lab 1 pipeline:")
        print("    python training/scripts/train_groot.py")
        sys.exit(1)
    
    arn = packages["ModelPackageSummaryList"][0]["ModelPackageArn"]
    print(f"  Latest model: {arn}")
    return arn


def pretrain_mlp_from_teleop():
    """
    EXPERIMENTAL / OPTIONAL — not part of the standard RL pipeline.
    Pre-train a lightweight MLP policy via behavioral cloning directly on the
    UR3 teleop dataset (the same demonstrations, NOT loaded from GR00T weights).
    Produces a .pt file that could seed RL. This is an experimental bridge, not a
    recognized Physical AI pattern: imitation (GR00T) and RL are separate pipelines.

    The MLP architecture matches what Isaac Lab expects:
    - Input: observation (state_dim)
    - Output: action (action_dim)
    - Hidden: 3 layers of 256 units with ELU activation
    """
    
    print("  Pre-training MLP policy from UR3 teleop data (behavioral cloning)...")
    print("  Learns directly from the teleop demonstrations — NOT loaded from GR00T weights")
    print("  Architecture: obs(7) → 256 → 256 → 256 → action(7)")
    
    # The actual pre-training would happen on SageMaker (needs GPU for speed).
    # For the pipeline, we create the training job:
    
    sm = boto3.client("sagemaker", region_name=REGION)
    
    job_name = f"groot-mlp-pretrain-{int(__import__('time').time())}"
    
    sm.create_training_job(
        TrainingJobName=job_name,
        RoleArn=ROLE_ARN,
        AlgorithmSpecification={
            "TrainingImage": ISAAC_LAB_IMAGE,
            "TrainingInputMode": "File",
        },
        InputDataConfig=[{
            "ChannelName": "training",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"s3://{BUCKET}/groot-data/ur3/dataset/data/",
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
        }],
        OutputDataConfig={
            "S3OutputPath": f"s3://{BUCKET}/mlp-pretrained/",
        },
        ResourceConfig={
            "InstanceType": "ml.g5.xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 50,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 600},
        HyperParameters={
            "mode": "pretrain_mlp",
            "state_dim": "7",
            "action_dim": "7",
            "hidden_dim": "256",
            "num_layers": "3",
            "epochs": "50",
            "lr": "0.001",
            "batch_size": "64",
        },
    )
    
    print(f"  Training job launched: {job_name}")
    print(f"  This trains an MLP to mimic the teleop data (~5 min)")
    return job_name


def launch_rl_refinement(pretrained_path: str = None):
    """
    Launch Isaac Lab RL refinement with pretrained MLP weights.
    
    If pretrained_path is provided, loads those weights as initialization.
    Otherwise trains from scratch (still works, just slower convergence).
    """
    
    sm = boto3.client("sagemaker", region_name=REGION)
    
    job_name = f"isaac-lab-rl-ur3-{int(__import__('time').time())}"
    
    hyperparams = {
        "task": "PickAndPlaceUR3-v0",
        "num_envs": "4096",
        "max_iterations": "50",
        "framework": "rsl_rl",
    }
    
    if pretrained_path:
        hyperparams["pretrained_model"] = pretrained_path
    
    sm.create_training_job(
        TrainingJobName=job_name,
        RoleArn=ROLE_ARN,
        AlgorithmSpecification={
            "TrainingImage": ISAAC_LAB_IMAGE,
            "TrainingInputMode": "File",
        },
        OutputDataConfig={
            "S3OutputPath": f"s3://{BUCKET}/isaac-lab/output/",
        },
        ResourceConfig={
            "InstanceType": "ml.g5.xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 100,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 1200},
        HyperParameters=hyperparams,
    )
    
    print(f"  RL refinement launched: {job_name}")
    print(f"  50 iterations, 4096 envs on ml.g5.xlarge (~5-10 min)")
    return job_name


def run_end_to_end():
    """
    Launch Isaac Lab RL training directly.

    This runs RL training from scratch without requiring a GR00T model.
    Imitation (GR00T) and RL (Isaac Lab) are separate approaches to obtaining
    a policy, not sequential stages.

    The policy learns pick-and-place through trial-and-error in simulation,
    guided by reward signals and domain randomization.
    """

    print("="*60)
    print("  Isaac Lab RL Training")
    print("="*60)

    # Launch RL training directly
    print("\n  Launching RL training on SageMaker...")
    rl_job = launch_rl_refinement()

    print(f"\n  RL training running!")
    print(f"  Monitor: aws sagemaker describe-training-job --training-job-name {rl_job} --region {REGION}")
    print(f"\n  Policy trains via PPO with domain randomization")
    print(f"  Output: S3 checkpoint ready for edge deployment (Lab 5)")

    return rl_job


def main():
    parser = argparse.ArgumentParser(description="GR00T → RL Bridge")
    parser.add_argument("action", choices=["end-to-end", "pretrain-mlp", "rl-refine", "status"],
                        help="Pipeline action")
    parser.add_argument("--pretrained", default=None, help="Path to pretrained MLP checkpoint")
    args = parser.parse_args()
    
    if args.action == "end-to-end":
        run_end_to_end()
    elif args.action == "pretrain-mlp":
        pretrain_mlp_from_teleop()
    elif args.action == "rl-refine":
        launch_rl_refinement(args.pretrained)
    elif args.action == "status":
        get_latest_model_package()


if __name__ == "__main__":
    main()
