#!/bin/bash
set -e
exec > /var/log/cosmos-bootstrap.log 2>&1
export DEBIAN_FRONTEND=noninteractive

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
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "=== Checking GPU ==="
nvidia-smi

echo "=== Logging into ECR (us-east-1) ==="
apt-get install -yq awscli
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 802782083985.dkr.ecr.us-east-1.amazonaws.com

echo "=== Pulling Cosmos container ==="
docker pull 802782083985.dkr.ecr.us-east-1.amazonaws.com/physical-ai/cosmos-transfer:latest

echo "=== Starting Cosmos NIM ==="
NGC_KEY=$(aws secretsmanager get-secret-value --secret-id physical-ai/ngc-api-key --region us-east-1 --query SecretString --output text)
docker run -d --gpus all --name cosmos -p 8000:8000 -e NGC_API_KEY=$NGC_KEY 802782083985.dkr.ecr.us-east-1.amazonaws.com/physical-ai/cosmos-transfer:latest

echo "DONE" > /var/log/cosmos-bootstrap.summary
