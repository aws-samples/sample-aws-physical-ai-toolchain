#!/bin/bash
# =============================================================================
# Run ON the Cosmos 3 p5.48xlarge instance (via SSM session)
#
# This script:
#   1. Verifies 8x H100 GPUs
#   2. Pulls the vLLM-Omni Cosmos3 container (~30GB)
#   3. Retrieves HF token from Secrets Manager
#   4. Starts the Cosmos3-Super server
#
# Usage (from SSM session on the instance):
#   bash /tmp/setup-cosmos3-server.sh
# =============================================================================

set -e

REGION="us-east-2"
ACCOUNT_ID="804152302157"

echo "=== Step 1: Verify GPUs ==="
nvidia-smi -L
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "Found $GPU_COUNT GPUs"
if [ "$GPU_COUNT" -ne 8 ]; then
    echo "WARNING: Expected 8 GPUs, found $GPU_COUNT"
fi

echo ""
echo "=== Step 2: Pull vLLM-Omni Cosmos3 container ==="
echo "This is ~30GB, takes 5-10 min on high-bandwidth instance..."
docker pull vllm/vllm-omni:cosmos3

echo ""
echo "=== Step 3: Get HF token from Secrets Manager ==="
HF_TOKEN=$(aws secretsmanager get-secret-value \
  --secret-id physical-ai/hf-token \
  --region $REGION \
  --query 'SecretString' --output text)

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: Could not retrieve HF token from Secrets Manager"
    echo "Create it with: aws secretsmanager create-secret --name physical-ai/hf-token --secret-string 'hf_...' --region $REGION"
    exit 1
fi
echo "HF token retrieved (${#HF_TOKEN} chars)"

echo ""
echo "=== Step 4: Start Cosmos3-Super vLLM server ==="
echo "Model download: ~128GB from HuggingFace (first time, 15-20 min)"
echo "Subsequent starts use cached weights."

docker run -d \
  --name cosmos3 \
  --gpus all \
  --ipc=host \
  --shm-size=64g \
  -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HF_HOME=/workspace/hf-cache \
  -v /opt/hf-cache:/workspace/hf-cache \
  vllm/vllm-omni:cosmos3 \
  bash -c "vllm serve nvidia/Cosmos3-Super --omni --cfg-parallel-size 2 --ulysses-degree 4 --use-hsdp --hsdp-shard-size 8 --init-timeout 2400 --stage-init-timeout 1800 --host 0.0.0.0 --port 8000"

echo ""
echo "=== Server starting (background) ==="
echo ""
echo "Monitor progress:"
echo "  docker logs -f cosmos3 2>&1 | grep -E '(download|ready|error|Worker)'"
echo ""
echo "Check if ready:"
echo "  curl -s http://localhost:8000/v1/models"
echo ""
echo "When ready, it returns: {\"data\":[{\"id\":\"nvidia/Cosmos3-Super\",...}]}"
