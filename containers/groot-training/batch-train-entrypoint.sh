#!/bin/bash
# =============================================================================
# GR00T N1.6 AWS Batch Training Entrypoint
#
# Mirrors the Isaac Lab batch entrypoint pattern:
#   - Reads hyperparameters from environment variables
#   - Launches training via torchrun (multi-GPU)
#   - Uploads checkpoints to S3 via boto3 after training
#
# Environment variables (set by job definition):
#   BASE_MODEL          - HuggingFace model name (default: nvidia/GR00T-N1.6-3B)
#   DATASET_PATH        - Local path to training data (default: /opt/ml/input/data/training)
#   MAX_STEPS           - Training steps (default: 5000)
#   BATCH_SIZE          - Global batch size (default: 8)
#   LEARNING_RATE       - Learning rate (default: 1e-4)
#   GRAD_ACCUM_STEPS    - Gradient accumulation steps (default: 4)
#   CHECKPOINT_BUCKET   - S3 bucket for checkpoint upload
#   CHECKPOINT_PREFIX   - S3 key prefix (default: checkpoints/groot/<job_id>)
#   HF_TOKEN            - HuggingFace token for model download
#   DATASET_S3_URI      - S3 URI to download dataset from
# =============================================================================

set -e

# --- Configuration from environment ---
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.6-3B}"
DATASET_PATH="${DATASET_PATH:-/opt/ml/input/data/training}"
DATASET_S3_URI="${DATASET_S3_URI:-}"
MAX_STEPS="${MAX_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
OUTPUT_DIR="/opt/ml/model"
JOB_ID="${AWS_BATCH_JOB_ID:-local}"

# --- Download dataset from S3 ---
if [ -n "$DATASET_S3_URI" ]; then
    echo "Downloading dataset from S3: $DATASET_S3_URI ..."
    mkdir -p "$DATASET_PATH"
    aws s3 sync "$DATASET_S3_URI" "$DATASET_PATH/"
    echo "Dataset ready at $DATASET_PATH"
    ls "$DATASET_PATH/meta/" 2>/dev/null || echo "WARNING: no meta/ directory found"
else
    echo "WARNING: DATASET_S3_URI not set. Training will fail unless data exists at $DATASET_PATH"
fi

# --- Count GPUs ---
NPROC=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")

# --- Print info ---
echo "============================================================"
echo "  GR00T N1.6 Fine-Tuning (AWS Batch)"
echo "  Base model:       $BASE_MODEL"
echo "  Dataset:          $DATASET_PATH"
echo "  Output:           $OUTPUT_DIR"
echo "  Max steps:        $MAX_STEPS"
echo "  Batch size:       $BATCH_SIZE"
echo "  Learning rate:    $LEARNING_RATE"
echo "  Grad accum:       $GRAD_ACCUM_STEPS"
echo "  GPUs:             $NPROC"
echo "  Job ID:           $JOB_ID"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

# --- Export HF token if set ---
if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# --- Launch training ---
# The train_entrypoint.py handles multi-GPU via torchrun internally
# (maybe_relaunch_with_torchrun). We set hyperparams via env vars that
# _hp() reads as SM_HP_* fallbacks.
export SM_MODEL_DIR="$OUTPUT_DIR"
export SM_CHANNEL_TRAINING="$DATASET_PATH"
export SM_HP_base_model="$BASE_MODEL"
export SM_HP_max_steps="$MAX_STEPS"
export SM_HP_batch_size="$BATCH_SIZE"
export SM_HP_learning_rate="$LEARNING_RATE"
export SM_HP_gradient_accumulation_steps="$GRAD_ACCUM_STEPS"

echo "Launching GR00T training..."
python3 /opt/ml/code/train_entrypoint.py

echo "Training complete."

# --- Upload checkpoints to S3 ---
if [ -n "${CHECKPOINT_BUCKET:-}" ]; then
    S3_PREFIX="${CHECKPOINT_PREFIX:-checkpoints/groot/${JOB_ID}}"
    echo "Uploading checkpoints to s3://${CHECKPOINT_BUCKET}/${S3_PREFIX}/ ..."
    aws s3 sync "$OUTPUT_DIR" "s3://${CHECKPOINT_BUCKET}/${S3_PREFIX}/" --region "${AWS_DEFAULT_REGION:-us-east-2}"
    echo "Done. Checkpoints uploaded to s3://${CHECKPOINT_BUCKET}/${S3_PREFIX}/"
else
    echo "WARNING: CHECKPOINT_BUCKET not set. Checkpoints will be lost when container exits."
fi
