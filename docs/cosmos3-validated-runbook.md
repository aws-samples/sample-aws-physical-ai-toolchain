# Cosmos 3 Validated Runbook

**Validated:** 2026-06-30
**Instance:** p5.48xlarge (8x H100 80GB) via EC2 Capacity Block
**Region:** us-east-1f
**Image:** `vllm/vllm-omni:cosmos3` (mirrored to ECR)
**Result:** V2V generation successful — 81-frame UR3 wrist camera clip → 189-frame 1280x720 photorealistic output

---

## Prerequisites (one-time setup)

### 1. Create ECR repository and mirror the vLLM-Omni image

The image is ~30GB. Use CodeBuild to mirror it (faster than local pull/push):

```bash
# Create ECR repo
aws ecr create-repository --repository-name vllm-omni --region us-east-1

# Create a CodeBuild project to pull from Docker Hub and push to ECR
# (see scripts/mirror-vllm-omni.sh for the full script)
bash scripts/mirror-vllm-omni.sh

# Monitor (takes ~15 min):
aws codebuild batch-get-builds --ids <BUILD_ID> --region us-east-1 \
  --query 'builds[0].buildStatus'

# Verify:
aws ecr describe-images --repository-name vllm-omni --region us-east-1 \
  --query 'imageDetails[*].{tags:imageTags,size:imageSizeInBytes}'
```

### 2. Create IAM role + instance profile

```bash
# Create role with EC2 trust
aws iam create-role --role-name cosmos-test-role \
  --assume-role-policy-document '{
    "Version":"2012-10-17",
    "Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]
  }'

# Attach policies
aws iam attach-role-policy --role-name cosmos-test-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam attach-role-policy --role-name cosmos-test-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
aws iam attach-role-policy --role-name cosmos-test-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Create instance profile
aws iam create-instance-profile --instance-profile-name cosmos-test-profile
aws iam add-role-to-instance-profile \
  --instance-profile-name cosmos-test-profile \
  --role-name cosmos-test-role
```

### 3. Create security group (outbound-only)

```bash
# In the default VPC (or your VPC) — only needs outbound for ECR/HF/S3 pulls
aws ec2 create-security-group --group-name cosmos-test-sg \
  --description "Cosmos test - outbound only" \
  --region us-east-1
# Note: default SG allows all outbound. No inbound rules needed (SSM, not SSH).
```

### 4. Store HuggingFace token in Secrets Manager

```bash
aws secretsmanager create-secret --name physical-ai/hf-token \
  --secret-string "hf_YOUR_TOKEN_HERE" --region us-east-1
```

### 5. Accept HuggingFace gated model licenses

You MUST accept BOTH of these on the HF account that owns the token:
- https://huggingface.co/nvidia/Cosmos-Guardrail1 → "Expand to review and access"
- https://huggingface.co/nvidia/Cosmos-1.0-Guardrail → "Expand to review and access"

(vLLM-Omni downloads BOTH at startup. If either is missing → 403 → server crashes.)

---

## Reserve GPU Capacity (Capacity Block)

P5 instances are rarely available on-demand. Use an EC2 Capacity Block:

```bash
# Check available blocks (minimum duration varies; typically 8-24 hr blocks)
aws ec2 describe-capacity-block-offerings \
  --instance-type p5.48xlarge \
  --capacity-duration-hours 24 \
  --instance-count 1 \
  --region us-east-1

# Purchase (note the CapacityBlockOfferingId and the AvailabilityZone)
aws ec2 purchase-capacity-block \
  --capacity-block-offering-id <OFFERING_ID> \
  --instance-platform Linux/UNIX \
  --region us-east-1 \
  --tag-specifications 'ResourceType=capacity-reservation,Tags=[{Key=Name,Value=cosmos3-test}]'

# Monitor — wait for "active" (starts at the scheduled time)
aws ec2 describe-capacity-reservations \
  --capacity-reservation-ids <CR_ID> \
  --region us-east-1 \
  --query 'CapacityReservations[0].State'
```

---

## Launch the Instance

Once the capacity block is `active`:

```bash
# Variables
REGION="us-east-1"
CR_ID="<your-capacity-reservation-id>"
BLOCK_AZ="us-east-1f"  # from the capacity block output

# Find a subnet in the block's AZ
SUBNET=$(aws ec2 describe-subnets \
  --filters "Name=availability-zone,Values=$BLOCK_AZ" \
  --query 'Subnets[0].SubnetId' --output text --region $REGION)

# Get the Deep Learning AMI (has NVIDIA drivers pre-installed)
AMI=$(aws ec2 describe-images --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
  --region $REGION \
  --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' --output text)

# Launch into the capacity block
aws ec2 run-instances \
  --image-id $AMI \
  --instance-type p5.48xlarge \
  --placement AvailabilityZone=$BLOCK_AZ \
  --security-group-ids <YOUR_SG_ID> \
  --iam-instance-profile Name=cosmos-test-profile \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3"}}]' \
  --instance-market-options '{"MarketType":"capacity-block"}' \
  --capacity-reservation-specification '{"CapacityReservationTarget":{"CapacityReservationId":"'$CR_ID'"}}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=cosmos3-server}]' \
  --region $REGION

# Wait for it to be running
aws ec2 wait instance-running --instance-ids <INSTANCE_ID> --region $REGION
```

**Key:** The `--instance-market-options '{"MarketType":"capacity-block"}'` flag is
required for Capacity Block instances. Without it you get "market type (purchasing)
option is not valid."

---

## Set Up the Server (via SSM — no SSH needed)

Wait ~2 min for SSM agent to register, then:

```bash
INSTANCE_ID="<your-instance-id>"

# Verify SSM is online
aws ssm describe-instance-information \
  --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
  --region us-east-1 --query 'InstanceInformationList[0].PingStatus'
# → "Online"

# Verify GPUs
aws ssm send-command --instance-ids $INSTANCE_ID \
  --document-name AWS-RunShellScript \
  --parameters '{"commands":["nvidia-smi -L"]}' \
  --region us-east-1

# Login to ECR
aws ssm send-command --instance-ids $INSTANCE_ID \
  --document-name AWS-RunShellScript \
  --parameters '{"commands":["aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 802782083985.dkr.ecr.us-east-1.amazonaws.com"]}' \
  --region us-east-1

# Pull the image (~5 min, 10GB from in-region ECR)
aws ssm send-command --instance-ids $INSTANCE_ID \
  --document-name AWS-RunShellScript \
  --parameters '{"commands":["docker pull 802782083985.dkr.ecr.us-east-1.amazonaws.com/vllm-omni:cosmos3"]}' \
  --timeout-seconds 600 --region us-east-1

# Start the vLLM-Omni server
# HF_TOKEN is needed for gated model downloads (Cosmos3-Super + Cosmos-Guardrail)
aws ssm send-command --instance-ids $INSTANCE_ID \
  --document-name AWS-RunShellScript \
  --parameters '{"commands":["docker run -d --name cosmos3 --gpus all --ipc=host --shm-size=64g -p 8000:8000 -e HF_TOKEN=<YOUR_HF_TOKEN> -e HF_HOME=/workspace/hf-cache 802782083985.dkr.ecr.us-east-1.amazonaws.com/vllm-omni:cosmos3 bash -c \"vllm serve nvidia/Cosmos3-Super --omni --cfg-parallel-size 2 --ulysses-degree 4 --use-hsdp --hsdp-shard-size 8 --init-timeout 2400 --stage-init-timeout 1800 --host 0.0.0.0 --port 8000\""]}' \
  --region us-east-1
```

### Wait for model loading (~15-20 min first time)

The server downloads ~128GB of model weights from HuggingFace on first start.
Monitor progress:

```bash
# Check download progress
aws ssm send-command --instance-ids $INSTANCE_ID \
  --document-name AWS-RunShellScript \
  --parameters '{"commands":["docker exec cosmos3 du -sh /workspace/hf-cache 2>/dev/null"]}' \
  --region us-east-1

# Check if workers are ready (look for "Worker N ready to receive requests")
aws ssm send-command --instance-ids $INSTANCE_ID \
  --document-name AWS-RunShellScript \
  --parameters '{"commands":["docker logs cosmos3 2>&1 | grep -i ready | tail -5"]}' \
  --region us-east-1

# Health check (returns model info when ready)
aws ssm send-command --instance-ids $INSTANCE_ID \
  --document-name AWS-RunShellScript \
  --parameters '{"commands":["curl -s http://localhost:8000/v1/models"]}' \
  --region us-east-1
```

Server is ready when `/v1/models` returns `nvidia/Cosmos3-Super` in the response.

---

## Generate V2V (the actual demo)

### Pull a real wrist camera clip from your dataset

```bash
aws ssm send-command --instance-ids $INSTANCE_ID \
  --document-name AWS-RunShellScript \
  --parameters '{"commands":["aws s3 cp s3://physical-ai-dev-datasets-802782083985/groot-data/ur3/dataset/videos/chunk-000/observation.images.wrist/episode_000000.mp4 /tmp/episode_000000.mp4 --region us-east-1 && docker cp /tmp/episode_000000.mp4 cosmos3:/tmp/episode_000000.mp4"]}' \
  --region us-east-1
```

### Post the V2V request

**Required parameters** (from vLLM-Omni Cosmos3-Super recipe docs):
- `size=1280x720` — output resolution
- `num_frames=189` — output frame count (at 24fps = ~8 sec)
- `fps=24` — output framerate
- `num_inference_steps=35` — diffusion steps
- `guidance_scale=6.0` — prompt adherence
- `flow_shift=10.0` — generation parameter
- `extra_params` with `condition_frame_indexes_vision=[0,1]` and `condition_video_keep="first"` — V2V conditioning

```bash
aws ssm send-command --instance-ids $INSTANCE_ID \
  --document-name AWS-RunShellScript \
  --parameters '{"commands":["docker exec cosmos3 curl -sS -X POST http://localhost:8000/v1/videos/sync -H \"Accept: video/mp4\" -F \"model=nvidia/Cosmos3-Super\" -F \"prompt=Dark matte table with subtle wear marks and fine scratches, colorful wooden blocks, overhead workshop lighting\" -F \"size=1280x720\" -F \"num_frames=189\" -F \"fps=24\" -F \"num_inference_steps=35\" -F \"guidance_scale=6.0\" -F \"max_sequence_length=4096\" -F \"flow_shift=10.0\" -F \"extra_params={\\\"condition_frame_indexes_vision\\\":[0,1],\\\"condition_video_keep\\\":\\\"first\\\"}\" -F \"seed=42\" -F \"input_reference=@/tmp/episode_000000.mp4;type=video/mp4\" -o /tmp/output_v2v.mp4 --max-time 600 -w \"\\nHTTP:%{http_code} SIZE:%{size_download}\""]}' \
  --timeout-seconds 900 --region us-east-1
```

**Generation takes ~5-8 min on 8x H100** (first request has JIT warmup). Subsequent
requests are faster (~2-3 min).

### Upload results to S3

```bash
aws ssm send-command --instance-ids $INSTANCE_ID \
  --document-name AWS-RunShellScript \
  --parameters '{"commands":["docker cp cosmos3:/tmp/output_v2v.mp4 /tmp/output_v2v.mp4 && aws s3 cp /tmp/output_v2v.mp4 s3://physical-ai-dev-datasets-802782083985/cosmos-samples/augmented_episode_000000.mp4 --region us-east-1 && aws s3 cp /tmp/episode_000000.mp4 s3://physical-ai-dev-datasets-802782083985/cosmos-samples/original_episode_000000.mp4 --region us-east-1 && echo DONE"]}' \
  --region us-east-1
```

### Download and compare

```bash
aws s3 cp s3://physical-ai-dev-datasets-802782083985/cosmos-samples/original_episode_000000.mp4 ./
aws s3 cp s3://physical-ai-dev-datasets-802782083985/cosmos-samples/augmented_episode_000000.mp4 ./
# Open both side by side — original (640x480, 5fps) vs augmented (1280x720, 24fps)
```

---

## Terminate

```bash
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-east-1
```

The capacity block expires automatically at its end time. Instance charges stop on termination.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `403 Cannot access gated repo nvidia/Cosmos-1.0-Guardrail` | HF token doesn't have access to the guardrail model | Accept license at https://huggingface.co/nvidia/Cosmos-1.0-Guardrail |
| `403 Cannot access gated repo nvidia/Cosmos-Guardrail1` | Same — different guardrail repo | Accept at https://huggingface.co/nvidia/Cosmos-Guardrail1 |
| `InsufficientInstanceCapacity` | No P5 available on-demand | Use Capacity Blocks (see above) |
| `market type (purchasing) option is not valid` | Missing `--instance-market-options '{"MarketType":"capacity-block"}'` when launching into a CB | Add the flag |
| `condition_frame_indexes_vision outside latent video: latent_frames=1` | Video too short or wrong resolution | Use `size=1280x720`, `num_frames=189`, `fps=24` |
| Download stalls at ~1GB on a shard | HuggingFace rate-limiting | `docker restart cosmos3` (cached shards are preserved) |
| SSM not registering | Instance profile missing SSM policy, or no outbound internet | Verify `AmazonSSMManagedInstanceCore` attached, subnet has NAT/IGW |

---

## Validated Output

- **Input:** `episode_000000.mp4` — 81 frames, 5fps, 640x480 (UR3 wrist camera, pick-and-place)
- **Output:** `augmented_episode_000000.mp4` — 189 frames, 24fps, 1280x720 (photorealistic V2V generation)
- **Prompt used:** `"A robot arm performing pick and place in a photorealistic industrial warehouse with fluorescent lighting and scratched metal surfaces"`
- **Time:** ~5-8 min generation (after model loaded)
- **Total time from launch to result:** ~25 min (boot + pull + model load + generate)
- **Cost:** Capacity Block $479 (13-hr block); actual usage ~30 min

---

## Prompting Tips for V2V

The V2V model takes the structure/motion from your input video and applies visual style
from the prompt. Best practice:

- **DO** describe the visual environment: lighting, materials, textures, background
- **DON'T** describe the action (the model already has that from the input video)
- **DO** keep it concrete and visual: "scratched metal table, fluorescent overhead light"
- **DON'T** give instructions: "add scratches" or "leave blocks unchanged"

Good prompts:
- `"Dark matte table with subtle wear marks and fine scratches, colorful wooden blocks, overhead workshop lighting"`
- `"Industrial factory floor, concrete, metal shelving in background, LED panel lighting"`
- `"Clean laboratory workbench, bright even lighting, white walls"`
- `"Aged workshop surface with oil stains, warm task lighting, tool rack visible"`

Bad prompts (instructional — model doesn't follow instructions well):
- `"Add scratches to the table but leave blocks the same"`
- `"Make it look more realistic"`
