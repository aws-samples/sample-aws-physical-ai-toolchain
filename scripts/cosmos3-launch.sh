#!/usr/bin/env bash
# Launch a Cosmos 3 generation server on EC2 p5.48xlarge via Capacity Block.
#
# This script automates all of Lab 3's infrastructure setup:
#   1. Creates IAM role + instance profile (if needed)
#   2. Purchases a Capacity Block (guarantees P5 capacity)
#   3. Waits for the block to activate
#   4. Launches the instance into the block
#   5. Pulls the vLLM-Omni image and starts the Cosmos 3 server
#   6. Waits for the server to be ready
#
# Prerequisites:
#   - AWS CLI configured with a profile that has EC2/IAM/SSM/ECR/S3 permissions
#   - vllm-omni:cosmos3 image mirrored to your ECR (run scripts/mirror-vllm-omni.sh first)
#   - HuggingFace token in Secrets Manager (physical-ai/hf-token)
#   - HF gated model licenses accepted (nvidia/Cosmos-Guardrail1 + nvidia/Cosmos-1.0-Guardrail)
#
# Usage:
#   bash scripts/cosmos3-launch.sh              # full deploy (purchases block + launches)
#   bash scripts/cosmos3-launch.sh --dry-run    # show what would happen, no AWS calls
#   bash scripts/cosmos3-launch.sh --status     # check server readiness on existing instance
#   bash scripts/cosmos3-launch.sh --generate "prompt text" # generate a video with the given prompt
#
# Output: prints the INSTANCE_ID when ready. Use it for generation:
#   bash scripts/cosmos3-launch.sh --generate "A UR3 robot arm picks up a red block..."

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
ROLE_NAME="cosmos-test-role"
PROFILE_NAME="cosmos-test-profile"
SG_NAME="cosmos-test-sg"
IMAGE="${REGISTRY}/vllm-omni:cosmos3"
HF_SECRET="physical-ai/hf-token"

# --- Parse args ---
DRY_RUN=false
STATUS_ONLY=false
GENERATE_PROMPT=""
INSTANCE_ID="${COSMOS_INSTANCE_ID:-}"

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --status) STATUS_ONLY=true ;;
    --generate) shift; GENERATE_PROMPT="$1"; shift ;;
  esac
done

echo "============================================================"
echo "  Cosmos 3 World Generation Server"
echo "  Region: $REGION  Account: $ACCOUNT"
echo "============================================================"

# --- Status check ---
if [ "$STATUS_ONLY" = true ] && [ -n "$INSTANCE_ID" ]; then
  echo "Checking instance $INSTANCE_ID..."
  aws ssm send-command --instance-ids "$INSTANCE_ID" --document-name AWS-RunShellScript \
    --parameters '{"commands":["curl -s http://localhost:8000/v1/models || echo NOT_READY"]}' \
    --region "$REGION" --output text
  exit 0
fi

# --- Generate ---
if [ -n "$GENERATE_PROMPT" ] && [ -n "$INSTANCE_ID" ]; then
  echo "Generating with prompt: $GENERATE_PROMPT"
  SEED=${COSMOS_SEED:-$RANDOM}
  OUTPUT="/tmp/cosmos_output_${SEED}.mp4"
  CMD="docker exec cosmos3 curl -sS -X POST http://localhost:8000/v1/videos/sync \
    -H 'Accept: video/mp4' \
    -F 'model=nvidia/Cosmos3-Super' \
    -F 'prompt=${GENERATE_PROMPT}' \
    -F 'size=1280x720' -F 'num_frames=189' -F 'fps=24' \
    -F 'num_inference_steps=35' -F 'guidance_scale=6.0' \
    -F 'max_sequence_length=4096' -F 'flow_shift=10.0' \
    -F 'extra_params={\"condition_frame_indexes_vision\":[0,1],\"condition_video_keep\":\"first\"}' \
    -F 'seed=${SEED}' \
    -F 'input_reference=@/tmp/episode_000000.mp4;type=video/mp4' \
    -o ${OUTPUT} -w '\\nHTTP:%{http_code}' --max-time 600"
  echo "  Seed: $SEED"
  echo "  (takes ~5-8 min)..."
  aws ssm send-command --instance-ids "$INSTANCE_ID" --document-name AWS-RunShellScript \
    --parameters "{\"commands\":[\"$CMD\"]}" --timeout-seconds 900 --region "$REGION" \
    --query 'Command.CommandId' --output text
  exit 0
fi

if [ "$DRY_RUN" = true ]; then
  echo "[DRY RUN] Would:"
  echo "  1. Create IAM role $ROLE_NAME + instance profile $PROFILE_NAME"
  echo "  2. Purchase a Capacity Block for p5.48xlarge"
  echo "  3. Launch instance into the block"
  echo "  4. Pull $IMAGE and start vLLM-Omni server"
  echo "  5. Wait ~20 min for model loading"
  echo ""
  echo "  Estimated cost: ~\$479 (Capacity Block, 13 hrs)"
  echo "  Image must be in ECR first: bash scripts/mirror-vllm-omni.sh"
  exit 0
fi

# --- Step 1: IAM (idempotent) ---
echo ""
echo "Step 1: IAM role + instance profile..."
aws iam create-role --role-name "$ROLE_NAME" \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
  2>/dev/null || true
aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore 2>/dev/null || true
aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly 2>/dev/null || true
aws iam attach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess 2>/dev/null || true
aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME" 2>/dev/null || true
aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE_NAME" 2>/dev/null || true
echo "  ✓ IAM ready"

# --- Step 2: Capacity Block ---
echo ""
echo "Step 2: Finding available Capacity Block..."
OFFERING=$(aws ec2 describe-capacity-block-offerings \
  --instance-type p5.48xlarge --capacity-duration-hours 24 --instance-count 1 \
  --region "$REGION" --query 'CapacityBlockOfferings[0]' --output json 2>/dev/null)

if [ -z "$OFFERING" ] || [ "$OFFERING" = "null" ]; then
  echo "  ERROR: No Capacity Block available in $REGION. Try later or another region."
  exit 1
fi

OFFERING_ID=$(echo "$OFFERING" | python3 -c "import json,sys; print(json.load(sys.stdin)['CapacityBlockOfferingId'])")
BLOCK_AZ=$(echo "$OFFERING" | python3 -c "import json,sys; print(json.load(sys.stdin)['AvailabilityZone'])")
START=$(echo "$OFFERING" | python3 -c "import json,sys; print(json.load(sys.stdin)['StartDate'])")
PRICE=$(echo "$OFFERING" | python3 -c "import json,sys; print(json.load(sys.stdin).get('UpfrontFee','unknown'))")

echo "  Found: $OFFERING_ID in $BLOCK_AZ"
echo "  Starts: $START"
echo "  Price: \$$PRICE"
read -p "  Purchase this block? (y/n): " CONFIRM
[ "$CONFIRM" != "y" ] && exit 0

CR_ID=$(aws ec2 purchase-capacity-block --capacity-block-offering-id "$OFFERING_ID" \
  --instance-platform Linux/UNIX --region "$REGION" \
  --tag-specifications 'ResourceType=capacity-reservation,Tags=[{Key=Name,Value=cosmos3},{Key=Project,Value=physical-ai}]' \
  --query 'CapacityReservation.CapacityReservationId' --output text)
echo "  ✓ Purchased: $CR_ID"

# --- Step 3: Wait for block activation ---
echo ""
echo "Step 3: Waiting for block to activate (scheduled → active)..."
while true; do
  STATE=$(aws ec2 describe-capacity-reservations --capacity-reservation-ids "$CR_ID" \
    --region "$REGION" --query 'CapacityReservations[0].State' --output text)
  echo "  State: $STATE ($(date +%H:%M:%S))"
  [ "$STATE" = "active" ] && break
  sleep 30
done
echo "  ✓ Block is ACTIVE"

# --- Step 4: Launch instance ---
echo ""
echo "Step 4: Launching p5.48xlarge..."
SUBNET=$(aws ec2 describe-subnets --filters "Name=availability-zone,Values=$BLOCK_AZ" \
  --query 'Subnets[0].SubnetId' --output text --region "$REGION")
AMI=$(aws ec2 describe-images --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
  --region "$REGION" --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' --output text)

# Security group (idempotent)
VPC=$(aws ec2 describe-subnets --subnet-ids "$SUBNET" --region "$REGION" --query 'Subnets[0].VpcId' --output text)
SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC" \
  --region "$REGION" --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
  SG_ID=$(aws ec2 create-security-group --group-name "$SG_NAME" --description "Cosmos test" \
    --vpc-id "$VPC" --region "$REGION" --query 'GroupId' --output text)
fi

INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI" --instance-type p5.48xlarge \
  --placement AvailabilityZone="$BLOCK_AZ" \
  --security-group-ids "$SG_ID" \
  --iam-instance-profile Name="$PROFILE_NAME" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3"}}]' \
  --instance-market-options '{"MarketType":"capacity-block"}' \
  --capacity-reservation-specification '{"CapacityReservationTarget":{"CapacityReservationId":"'$CR_ID'"}}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=cosmos3-server},{Key=Project,Value=physical-ai}]' \
  --region "$REGION" --query 'Instances[0].InstanceId' --output text)

echo "  Instance: $INSTANCE_ID"
echo "  Waiting for running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
echo "  ✓ Running"

# --- Step 5: Setup server via SSM ---
echo ""
echo "Step 5: Setting up Cosmos 3 server (via SSM)..."
echo "  Waiting for SSM agent..."
while true; do
  PING=$(aws ssm describe-instance-information --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
    --region "$REGION" --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || echo "None")
  [ "$PING" = "Online" ] && break
  sleep 10
done
echo "  ✓ SSM online"

HF_TOKEN=$(aws secretsmanager get-secret-value --secret-id "$HF_SECRET" \
  --query SecretString --output text --region "$REGION")

echo "  ECR login..."
aws ssm send-command --instance-ids "$INSTANCE_ID" --document-name AWS-RunShellScript \
  --parameters '{"commands":["aws ecr get-login-password --region '$REGION' | docker login --username AWS --password-stdin '$REGISTRY'"]}' \
  --region "$REGION" > /dev/null
sleep 10

echo "  Pulling image (~5 min)..."
aws ssm send-command --instance-ids "$INSTANCE_ID" --document-name AWS-RunShellScript \
  --parameters '{"commands":["docker pull '$IMAGE'"]}' \
  --timeout-seconds 600 --region "$REGION" > /dev/null

echo "  Waiting for pull to complete..."
sleep 300

echo "  Starting vLLM-Omni server..."
aws ssm send-command --instance-ids "$INSTANCE_ID" --document-name AWS-RunShellScript \
  --parameters '{"commands":["docker run -d --name cosmos3 --gpus all --ipc=host --shm-size=64g -p 8000:8000 -e HF_TOKEN='$HF_TOKEN' -e HF_HOME=/workspace/hf-cache '$IMAGE' bash -c \"vllm serve nvidia/Cosmos3-Super --omni --cfg-parallel-size 2 --ulysses-degree 4 --use-hsdp --hsdp-shard-size 8 --init-timeout 2400 --stage-init-timeout 1800 --host 0.0.0.0 --port 8000\""]}' \
  --region "$REGION" > /dev/null

# --- Step 6: Wait for model loading ---
echo ""
echo "Step 6: Model loading (~15-20 min on first start)..."
while true; do
  HEALTH=$(aws ssm send-command --instance-ids "$INSTANCE_ID" --document-name AWS-RunShellScript \
    --parameters '{"commands":["curl -s http://localhost:8000/v1/models 2>/dev/null || echo NOT_READY"]}' \
    --region "$REGION" --query 'Command.CommandId' --output text)
  sleep 15
  RESULT=$(aws ssm get-command-invocation --command-id "$HEALTH" --instance-id "$INSTANCE_ID" \
    --region "$REGION" --query 'StandardOutputContent' --output text 2>/dev/null || echo "")
  if echo "$RESULT" | grep -q "Cosmos3-Super"; then
    break
  fi
  echo "  Still loading... ($(date +%H:%M:%S))"
  sleep 60
done

echo ""
echo "============================================================"
echo "  ✅ COSMOS 3 SERVER IS READY"
echo ""
echo "  Instance: $INSTANCE_ID"
echo "  To generate:"
echo "    export COSMOS_INSTANCE_ID=$INSTANCE_ID"
echo "    bash scripts/cosmos3-launch.sh --generate \"your prompt here\""
echo ""
echo "  To terminate when done:"
echo "    aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
echo "============================================================"
