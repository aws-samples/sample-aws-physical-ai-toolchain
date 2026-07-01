#!/usr/bin/env bash
# Mirror vllm/vllm-omni:cosmos3 to our ECR via CodeBuild.
# Run from anywhere — creates a one-off CodeBuild project, starts a build, then you can
# delete the project after the image is pushed.
set -euo pipefail

REGION="us-east-1"
ACCOUNT="802782083985"
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
REPO="vllm-omni"
TAG="cosmos3"
PROJECT_NAME="cosmos3-mirror-vllm-omni"
ROLE="arn:aws:iam::${ACCOUNT}:role/physical-ai-dev-codebuild-role"

echo "=== Mirroring vllm/vllm-omni:cosmos3 → ${REGISTRY}/${REPO}:${TAG} ==="

# Create project (idempotent)
aws codebuild create-project \
  --name "${PROJECT_NAME}" \
  --source '{
    "type": "NO_SOURCE",
    "buildspec": "version: 0.2\nphases:\n  build:\n    commands:\n      - aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 802782083985.dkr.ecr.us-east-1.amazonaws.com\n      - docker pull vllm/vllm-omni:cosmos3\n      - docker tag vllm/vllm-omni:cosmos3 802782083985.dkr.ecr.us-east-1.amazonaws.com/vllm-omni:cosmos3\n      - docker push 802782083985.dkr.ecr.us-east-1.amazonaws.com/vllm-omni:cosmos3\n"
  }' \
  --artifacts '{"type": "NO_ARTIFACTS"}' \
  --environment '{
    "type": "LINUX_CONTAINER",
    "image": "aws/codebuild/standard:7.0",
    "computeType": "BUILD_GENERAL1_LARGE",
    "privilegedMode": true
  }' \
  --service-role "${ROLE}" \
  --region "${REGION}" 2>/dev/null || echo "  (project already exists)"

# Start build
BUILD_ID=$(aws codebuild start-build --project-name "${PROJECT_NAME}" --region "${REGION}" \
  --query 'build.id' --output text)
echo "  Build started: ${BUILD_ID}"
echo "  Monitor: aws codebuild batch-get-builds --ids ${BUILD_ID} --region ${REGION} --query 'builds[0].buildStatus'"
echo ""
echo "  This pulls ~30 GB and pushes to ECR. Takes 10-20 min on BUILD_GENERAL1_LARGE."
echo "  Once done, image will be at: ${REGISTRY}/${REPO}:${TAG}"
