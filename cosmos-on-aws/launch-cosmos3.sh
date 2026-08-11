#!/bin/bash
# =============================================================================
# Cosmos 3 — Launch & Setup Script
#
# Prerequisites:
#   - An ACTIVE P5 Capacity Block (see ec2-deployment-guide.md Step 1)
#   - HF token stored in Secrets Manager: physical-ai/hf-token
#   - ECR repo exists: physical-ai/cosmos3
#
# BEFORE RUNNING: replace every <PLACEHOLDER> below with your own
# account/environment values (see ec2-deployment-guide.md Step 2). SUBNET,
# SG, and PROFILE come from the Foundation stack's Terraform outputs; AMI is
# region-specific (latest AL2 GPU-optimized ECS/EKS AMI or Deep Learning AMI).
#
# Usage:
#   bash cosmos-on-aws/launch-cosmos3.sh
# =============================================================================

set -e

REGION="<REGION>"
AZ="<AVAILABILITY_ZONE>"
CR_ID="<CAPACITY_RESERVATION_ID>"
SUBNET="<SUBNET_ID>"
SG="<SECURITY_GROUP_ID>"
PROFILE="<INSTANCE_PROFILE_NAME>"
AMI="<AMI_ID>"
INSTANCE_TYPE="p5.48xlarge"

echo "=== Checking Capacity Block state ==="
STATE=$(aws ec2 describe-capacity-reservations \
  --capacity-reservation-ids $CR_ID \
  --region $REGION \
  --query 'CapacityReservations[0].State' --output text)

echo "Capacity Block state: $STATE"
if [ "$STATE" != "active" ]; then
    echo "ERROR: Capacity Block is not yet active (state: $STATE)"
    echo "Wait until the start time and try again."
    exit 1
fi

echo ""
echo "=== Launching p5.48xlarge into Capacity Block ==="
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $AMI \
  --instance-type $INSTANCE_TYPE \
  --placement AvailabilityZone=$AZ \
  --subnet-id $SUBNET \
  --security-group-ids $SG \
  --iam-instance-profile Name=$PROFILE \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3","Encrypted":true}}]' \
  --instance-market-options '{"MarketType":"capacity-block"}' \
  --capacity-reservation-specification '{"CapacityReservationTarget":{"CapacityReservationId":"'$CR_ID'"}}' \
  --metadata-options '{"HttpTokens":"required"}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=cosmos3-server},{Key=Project,Value=physical-ai},{Key=Environment,Value=dev}]' \
  --region $REGION \
  --query 'Instances[0].InstanceId' --output text)

echo "Instance launched: $INSTANCE_ID"
echo ""
echo "=== Waiting for instance to be running ==="
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION
echo "Instance is running."

echo ""
echo "=== Waiting for SSM to register (up to 3 min) ==="
for i in $(seq 1 18); do
  STATUS=$(aws ssm describe-instance-information \
    --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
    --region $REGION \
    --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null)
  if [ "$STATUS" = "Online" ]; then
    echo "SSM is online."
    break
  fi
  echo "  Waiting... ($i/18)"
  sleep 10
done

if [ "$STATUS" != "Online" ]; then
  echo "WARNING: SSM not online yet. Instance may need more time."
  echo "Instance ID: $INSTANCE_ID"
  exit 1
fi

echo ""
echo "============================================================"
echo "  Cosmos 3 instance ready!"
echo "  Instance ID: $INSTANCE_ID"
echo "  Region:      $REGION"
echo "  AZ:          $AZ"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Create HF token secret (if not done):"
echo "     aws secretsmanager create-secret --name physical-ai/hf-token --secret-string 'hf_YOUR_TOKEN' --region $REGION"
echo ""
echo "  2. Run the setup commands via SSM:"
echo "     aws ssm start-session --target $INSTANCE_ID --region $REGION"
echo ""
echo "  3. Inside the instance:"
echo "     docker pull vllm/vllm-omni:cosmos3"
echo "     # Then start the server (see cosmos-on-aws/README.md)"
