#!/bin/bash
# =============================================================================
# Edge deployment orchestrator (called by the OSMO workflow's edge stage and
# usable standalone). Chains: export trained checkpoint -> publish Greengrass
# component -> deploy to the robot fleet.
#
# Nothing is hardcoded: the scripts resolve account/region/buckets from the
# caller. Pass --dry-run to preview the whole chain with no AWS writes.
#
# Usage:
#   edge/deploy.sh --checkpoint s3://.../model.tar.gz --dry-run
#   edge/deploy.sh --checkpoint ./model.pt --inference-version 1.0.1
# =============================================================================
set -euo pipefail

CHECKPOINT=""
INFERENCE_VERSION="1.0.0"
DRY_RUN=""
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

while [ $# -gt 0 ]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --checkpoint=*) CHECKPOINT="${1#*=}"; shift ;;
    --inference-version) INFERENCE_VERSION="$2"; shift 2 ;;
    --dry-run) DRY_RUN="--dry-run"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [ -z "$CHECKPOINT" ]; then
  echo "ERROR: --checkpoint is required (TorchScript .pt or s3:// model.tar.gz)"
  exit 1
fi

echo "=== [1/3] Export checkpoint -> ONNX + TensorRT ==="
# export.py has no --dry-run and needs a GPU/TensorRT to compile; in dry-run we
# only print what would run (it also requires a real TorchScript checkpoint).
if [ -n "$DRY_RUN" ]; then
  echo "[dry-run] would run: export.py --checkpoint $CHECKPOINT \\"
  echo "            --output-onnx model_exported/policy.onnx \\"
  echo "            --output-trt model_exported/policy.trt --target-device jetson-orin --fp16"
else
  python3 "$REPO_ROOT/training/scripts/export.py" \
    --checkpoint "$CHECKPOINT" \
    --output-onnx "$REPO_ROOT/model_exported/policy.onnx" \
    --output-trt "$REPO_ROOT/model_exported/policy.trt" \
    --target-device jetson-orin --fp16
fi

echo "=== [2/3] Publish Greengrass inference component ==="
python3 "$REPO_ROOT/edge/create_component.py" \
  --model "$REPO_ROOT/model_exported/policy.trt" \
  --component-version "$INFERENCE_VERSION" $DRY_RUN

echo "=== [3/3] Deploy component to the robot fleet ==="
python3 "$REPO_ROOT/edge/deploy_to_fleet.py" \
  --inference-version "$INFERENCE_VERSION" $DRY_RUN

echo "=== Done. ==="
