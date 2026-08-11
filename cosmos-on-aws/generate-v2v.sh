#!/bin/bash
# =============================================================================
# Cosmos 3 — V2V Generation Script
#
# Run ON the Cosmos 3 instance after the server is ready.
# Generates a V2V (video-to-video) augmentation from a UR3 wrist camera clip.
#
# Usage:
#   bash /tmp/generate-v2v.sh
# =============================================================================

set -e

# Account-specific — replace both before running (see ec2-deployment-guide.md).
# Or set DATASETS_BUCKET directly, e.g. from:
#   terraform -chdir=../foundation/infra output -raw datasets_bucket_name
REGION="<REGION>"
ACCOUNT_ID="<ACCOUNT_ID>"
DATASETS_BUCKET="physical-ai-dev-datasets-${ACCOUNT_ID}"

echo "=== Step 1: Verify server is ready ==="
MODELS=$(curl -s http://localhost:8000/v1/models 2>/dev/null)
if echo "$MODELS" | grep -q "Cosmos3-Super"; then
    echo "✅ Server ready: nvidia/Cosmos3-Super"
else
    echo "❌ Server not ready. Check: docker logs cosmos3 2>&1 | tail -20"
    exit 1
fi

echo ""
echo "=== Step 2: Download input video from S3 ==="
aws s3 cp s3://${DATASETS_BUCKET}/groot-data/ur3/dataset/videos/chunk-000/observation.images.wrist/episode_000000.mp4 \
  /tmp/episode_000000.mp4 --region $REGION

echo "Input video:"
ls -lh /tmp/episode_000000.mp4

echo ""
echo "=== Step 3: Copy video into container ==="
docker cp /tmp/episode_000000.mp4 cosmos3:/tmp/episode_000000.mp4

echo ""
echo "=== Step 4: Generate V2V (this takes 5-8 min) ==="
PROMPT="A UR3 robot arm with a Robotiq gripper reaches down to a dark matte table, grasps a small red wooden block, lifts it slowly, and places it onto a yellow sticky note approximately 6 inches away. Top-down wrist camera view. Colorful wooden blocks are scattered on the table. Smooth deliberate motion."

echo "Prompt: $PROMPT"
echo ""
echo "Generating 189 frames at 1280x720, 24fps..."
echo "Start time: $(date -u)"

docker exec cosmos3 curl -sS -X POST http://localhost:8000/v1/videos/sync \
  -H "Accept: video/mp4" \
  -F "model=nvidia/Cosmos3-Super" \
  -F "prompt=${PROMPT}" \
  -F "size=1280x720" \
  -F "num_frames=189" \
  -F "fps=24" \
  -F "num_inference_steps=35" \
  -F "guidance_scale=6.0" \
  -F "max_sequence_length=4096" \
  -F "flow_shift=10.0" \
  -F "extra_params={\"condition_frame_indexes_vision\":[0,1],\"condition_video_keep\":\"first\"}" \
  -F "seed=100" \
  -F "input_reference=@/tmp/episode_000000.mp4;type=video/mp4" \
  -o /tmp/output_v2v.mp4 \
  --max-time 600 \
  -w "\nHTTP:%{http_code} SIZE:%{size_download}\n"

echo "End time: $(date -u)"
echo ""

echo "=== Step 5: Verify output ==="
docker cp cosmos3:/tmp/output_v2v.mp4 /tmp/output_v2v.mp4
ls -lh /tmp/output_v2v.mp4

echo ""
echo "=== Step 6: Upload to S3 ==="
aws s3 cp /tmp/output_v2v.mp4 \
  s3://${DATASETS_BUCKET}/cosmos-samples/augmented_episode_000000.mp4 \
  --region $REGION

aws s3 cp /tmp/episode_000000.mp4 \
  s3://${DATASETS_BUCKET}/cosmos-samples/original_episode_000000.mp4 \
  --region $REGION

echo ""
echo "============================================================"
echo "  ✅ V2V Generation Complete!"
echo ""
echo "  Original: s3://${DATASETS_BUCKET}/cosmos-samples/original_episode_000000.mp4"
echo "  Generated: s3://${DATASETS_BUCKET}/cosmos-samples/augmented_episode_000000.mp4"
echo ""
echo "  Download and compare:"
echo "    aws s3 cp s3://${DATASETS_BUCKET}/cosmos-samples/ ./cosmos-output/ --recursive --region $REGION"
echo "============================================================"
