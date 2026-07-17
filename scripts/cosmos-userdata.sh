#!/bin/bash
set -e
exec > /var/log/cosmos-bootstrap.log 2>&1
export DEBIAN_FRONTEND=noninteractive

# Cosmos Transfer 2.5 EC2 bootstrap — the VALIDATED way to run Cosmos (the
# SageMaker endpoint path does not work: SageMaker GPUs ship driver 470, Cosmos
# needs 580+). Runs the NIM container serving /v1/infer on port 8000.
#
# Requires a p5 (8x H100) Spot instance. Everything is derived from the instance's
# own account/region — nothing hardcoded. See docs/cosmos-deployment-guide.md.
#
# Override defaults by exporting before first boot, or edit here:
PROJECT_NAME="${PROJECT_NAME:-physical-ai}"
NGC_SECRET_NAME="${NGC_SECRET_NAME:-physical-ai/ngc-api-key}"

echo "=== Installing SSM agent ==="
snap install amazon-ssm-agent --classic
systemctl enable snap.amazon-ssm-agent.amazon-ssm-agent.service
systemctl start snap.amazon-ssm-agent.amazon-ssm-agent.service

echo "=== Installing Docker ==="
curl -fsSL https://get.docker.com | sh
systemctl enable docker && systemctl start docker

echo "=== Installing NVIDIA Container Toolkit ==="
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
apt-get update -yq && apt-get install -yq nvidia-container-toolkit

# Docker GPU runtime must be the DEFAULT or `--gpus all` fails with
# "Error 802: system not yet initialized" (per the deployment runbook).
nvidia-ctk runtime configure --runtime=docker
cat > /etc/docker/daemon.json <<'DAEMON'
{"default-runtime": "nvidia", "runtimes": {"nvidia": {"args": [], "path": "nvidia-container-runtime"}}}
DAEMON
systemctl restart docker

# p5 uses NVSwitch — CUDA won't initialize without fabricmanager.
echo "=== Installing nvidia-fabricmanager (required on p5) ==="
apt-get install -yq nvidia-fabricmanager-550 || echo "fabricmanager install skipped (not a NVSwitch box?)"
systemctl enable nvidia-fabricmanager 2>/dev/null || true
systemctl start nvidia-fabricmanager 2>/dev/null || true

echo "=== Checking GPU ==="
nvidia-smi

echo "=== Resolving account / region ==="
apt-get install -yq awscli
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 60" || true)
REGION=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/placement/region)
REGION="${REGION:-$(aws configure get region)}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
COSMOS_IMAGE="${ECR_REGISTRY}/${PROJECT_NAME}/cosmos-transfer:latest"
echo "    Region:  $REGION"
echo "    Account: $ACCOUNT_ID"
echo "    Image:   $COSMOS_IMAGE"

echo "=== Logging into ECR + pulling Cosmos container ==="
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY"
docker pull "$COSMOS_IMAGE"

echo "=== Starting Cosmos NIM (port 8000, /v1/infer) ==="
NGC_KEY=$(aws secretsmanager get-secret-value --secret-id "$NGC_SECRET_NAME" --region "$REGION" --query SecretString --output text)
# Docker flags + NIM profile per the runbook: --ipc=host + ulimits for multi-GPU,
# NIM_MODEL_PROFILE=latency to use all 8 H100s (CP=8) instead of CP=1.
docker run -d --gpus all --name cosmos \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 \
  -e NGC_API_KEY="$NGC_KEY" \
  -e NIM_MODEL_PROFILE=latency \
  "$COSMOS_IMAGE"

echo "=== Waiting for health endpoint ==="
for i in $(seq 1 60); do
  if curl -sf http://localhost:8000/v1/health/ready >/dev/null 2>&1; then
    echo "Cosmos NIM ready"; break
  fi
  sleep 10
done

echo "DONE" > /var/log/cosmos-bootstrap.summary
