#!/bin/bash
# =============================================================================
# AWS Physical AI Toolchain — Path A Quick Start
#
# Runs the complete GR00T fine-tuning pipeline on SageMaker:
#   1. Reads CDK outputs (bucket, role, ECR URI)
#   2. Uploads demo dataset to S3
#   3. Launches SageMaker training job
#
# Prerequisites:
#   - CDK Foundation stack deployed (cdk deploy --context mode=simple)
#   - Training container pushed to ECR
#   - Demo dataset downloaded (or provide your own)
#   - HF_TOKEN set (for model download during training)
#
# Usage:
#   ./run-path-a.sh                                # Use the bundled UR3 dataset
#   ./run-path-a.sh --dataset-dir ./my-data        # Use your own data (space or = form)
#   ./run-path-a.sh --dataset-dir=./my-data --dry-run
#   ./run-path-a.sh --dry-run                      # Show what would happen
# =============================================================================

set -euo pipefail

STACK_NAME="PhysicalAi-dev-Foundation"
DATASET_DIR="training/data/ur3_lerobot_dataset"   # the dataset that ships in this repo
PREFIX="groot-data/ur3"                            # matches Lab 1 + pipeline.py defaults
MAX_STEPS=5000
DRY_RUN=""

# Parse flags. Supports both "--flag value" and "--flag=value" forms, and a bare
# positional dataset dir. (A bare "--dry-run" must NOT be captured as the dataset
# path — that was the old bug.)
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)        DRY_RUN="--dry-run" ;;
    --dataset-dir)    DATASET_DIR="$2"; shift ;;
    --dataset-dir=*)  DATASET_DIR="${1#*=}" ;;
    --max-steps)      MAX_STEPS="$2"; shift ;;
    --max-steps=*)    MAX_STEPS="${1#*=}" ;;
    --*)              echo "Unknown flag: $1" >&2; exit 2 ;;
    *)                DATASET_DIR="$1" ;;   # bare positional dataset dir
  esac
  shift
done

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║   AWS Physical AI Toolchain — Path A    ║"
echo "  ║   GR00T Fine-Tuning on SageMaker        ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""

# --- Read CDK Outputs ---
echo "  [1/4] Reading infrastructure outputs..."
BUCKET=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" \
  --query 'Stacks[0].Outputs[?OutputKey==`DatasetsBucketName`].OutputValue' --output text)
ROLE_ARN=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" \
  --query 'Stacks[0].Outputs[?OutputKey==`SageMakerRoleArn`].OutputValue' --output text)
ECR_URI=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" \
  --query 'Stacks[0].Outputs[?OutputKey==`GrootTrainingRepoUri`].OutputValue' --output text)

echo "    Bucket:   $BUCKET"
echo "    Role:     $ROLE_ARN"
echo "    ECR:      $ECR_URI"
echo ""

# --- Check dataset exists ---
echo "  [2/4] Checking dataset..."
if [ ! -d "$DATASET_DIR" ]; then
  echo "    Dataset not found at: $DATASET_DIR"
  echo "    Download with: python training/groot/download_demo_dataset.py --output $DATASET_DIR"
  exit 1
fi
echo "    Found: $DATASET_DIR"
echo ""

# --- Upload to S3 ---
echo "  [3/4] Uploading dataset to S3..."
aws s3 sync "$DATASET_DIR" "s3://$BUCKET/$PREFIX/dataset/" --quiet
echo "    Uploaded to: s3://$BUCKET/$PREFIX/dataset/"
echo ""

# --- Launch Training ---
echo "  [4/4] Launching SageMaker training job..."
python training/groot/launch_training.py \
  --s3-bucket "$BUCKET" \
  --dataset-prefix "$PREFIX" \
  --role-arn "$ROLE_ARN" \
  --ecr-image "$ECR_URI:latest" \
  --max-steps "$MAX_STEPS" \
  $DRY_RUN

echo ""
echo "  Done! Monitor training with:"
echo "    aws sagemaker list-training-jobs --sort-by CreationTime --sort-order Descending --max-results 1"
echo ""
