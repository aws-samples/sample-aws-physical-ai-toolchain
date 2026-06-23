#!/usr/bin/env bash
################################################################################
# deploy-workstation.sh
#
# Deploys the Isaac Sim GPU workstation with automatic AZ retry on capacity failure.
#
# NOTE: This workstation uses the NVIDIA Isaac Sim Marketplace AMI (product ID
# prodview-bl35herdyozhw, https://aws.amazon.com/marketplace/pp/prodview-bl35herdyozhw).
# You MUST subscribe to this product in the AWS Marketplace before running this script,
# or the EC2 launch will fail with a subscription required error.
#
# PROBLEM: CloudFormation is declarative and cannot retry EC2 launches across AZs.
# When deploying a GPU instance (e.g., g6e.4xlarge), a given AZ may have no capacity
# (InsufficientInstanceCapacity error). Different AZs in the same region often have
# different availability.
#
# SOLUTION: This script wraps `cdk deploy` and automatically retries in different
# AZs when capacity errors occur. It:
#   1. Queries which AZs offer the requested instance type in the target region
#   2. Attempts deployment in the first AZ
#   3. On InsufficientInstanceCapacity, deletes the failed stack and tries the next AZ
#   4. Stops on non-capacity errors (misconfiguration, IAM issues, etc.)
#   5. Exits successfully when any AZ succeeds
#
# USAGE:
#   ./deploy-workstation.sh
#
#   # Override instance type and allowed CIDR:
#   INSTANCE_TYPE=g6e.8xlarge ALLOWED_CIDR=203.0.113.5/32 ./deploy-workstation.sh
#
#   # Override stack name (for multiple workstations):
#   STACK_NAME=PhysicalAi-dev-Workstation-TeamA ./deploy-workstation.sh
#
# ENVIRONMENT VARIABLES:
#   AWS_REGION       AWS region (default: from aws configure or us-east-1)
#   INSTANCE_TYPE    GPU instance type (default: g6e.4xlarge)
#   ALLOWED_CIDR     CIDR allowed to connect via DCV (default: auto-detect your IP)
#   STACK_NAME       CloudFormation stack name (default: PhysicalAi-dev-Workstation)
#
################################################################################

set -uo pipefail  # Exit on unset variables and pipe failures (NOT -e, we need to catch deploy failures)

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

REGION="${AWS_REGION:-$(aws configure get region 2>/dev/null || echo us-east-1)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g6e.4xlarge}"
STACK_NAME="${STACK_NAME:-PhysicalAi-dev-Workstation}"

# Auto-detect caller IP if ALLOWED_CIDR not set
if [[ -z "${ALLOWED_CIDR:-}" ]]; then
  echo "==> Detecting your public IP for DCV access..."
  CALLER_IP=$(curl -s --max-time 5 ifconfig.me || echo "")
  if [[ -n "$CALLER_IP" ]]; then
    ALLOWED_CIDR="${CALLER_IP}/32"
    echo "    Detected IP: $CALLER_IP (using $ALLOWED_CIDR)"
    echo ""
    echo "    NOTE: This is the deploy host's egress IP. If DCV connection fails from your"
    echo "    browser, you may need to allow-list your browser machine's IP instead:"
    echo "      ALLOWED_CIDR=<your-browser-ip>/32 $0"
  else
    echo ""
    echo "ERROR: Could not auto-detect your public IP (ifconfig.me timed out or failed)."
    echo "       For security, this script will NOT default to 0.0.0.0/0 (open to world)."
    echo ""
    echo "Please set ALLOWED_CIDR explicitly:"
    echo "  ALLOWED_CIDR=YOUR_IP/32 $0"
    echo ""
    echo "To check your IP: curl ifconfig.me"
    echo "To explicitly allow all (not recommended): ALLOWED_CIDR=0.0.0.0/0 $0"
    echo ""
    exit 1
  fi
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo " Deploying Isaac Sim Workstation"
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  Region:         $REGION"
echo "  Instance Type:  $INSTANCE_TYPE"
echo "  Allowed CIDR:   $ALLOWED_CIDR  (DCV port 8443 + SSH port 22 will accept from this CIDR)"
echo "  Stack Name:     $STACK_NAME"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Change to the cdk directory (script can be run from anywhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ═══════════════════════════════════════════════════════════════════════════════
# Enumerate AZs that offer the instance type
# ═══════════════════════════════════════════════════════════════════════════════

echo "==> Querying AZs that offer $INSTANCE_TYPE in $REGION..."
AZ_LIST=$(aws ec2 describe-instance-type-offerings \
  --location-type availability-zone \
  --filters "Name=instance-type,Values=$INSTANCE_TYPE" \
  --region "$REGION" \
  --query 'InstanceTypeOfferings[].Location' \
  --output text)

if [[ -z "$AZ_LIST" ]]; then
  echo "ERROR: No AZs in $REGION offer instance type $INSTANCE_TYPE."
  echo "       Check that the instance type is available in this region."
  exit 1
fi

# Convert space-separated string to array
read -ra AZS <<< "$AZ_LIST"
echo "    Found ${#AZS[@]} AZ(s): ${AZS[*]}"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Check if stack exists and get its status
# ═══════════════════════════════════════════════════════════════════════════════

get_stack_status() {
  aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query 'Stacks[0].StackStatus' \
    --output text 2>/dev/null || echo "DOES_NOT_EXIST"
}

# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Check if deploy output indicates a capacity error
# ═══════════════════════════════════════════════════════════════════════════════

is_capacity_error() {
  local output="$1"
  # BUG FIX 3: Tighten capacity-error detection to avoid false positives
  # Match specific EC2 capacity error signals only
  if echo "$output" | grep -q "InsufficientInstanceCapacity"; then
    return 0
  fi
  # Match the exact phrase patterns EC2 emits for capacity issues
  if echo "$output" | grep -q "do not have sufficient .* capacity"; then
    return 0
  fi
  if echo "$output" | grep -q "currently do not have sufficient"; then
    return 0
  fi
  return 1
}

# ═══════════════════════════════════════════════════════════════════════════════
# Iterate over AZs and attempt deployment
# ═══════════════════════════════════════════════════════════════════════════════

ATTEMPT=0
for AZ in "${AZS[@]}"; do
  ATTEMPT=$((ATTEMPT + 1))
  echo "────────────────────────────────────────────────────────────────────────────────"
  echo "Attempt $ATTEMPT/${#AZS[@]}: Deploying to AZ $AZ"
  echo "────────────────────────────────────────────────────────────────────────────────"

  # BUG FIX 1: Reset exit code at start of each iteration to avoid stale failures
  unset DEPLOY_EXIT_CODE

  # Run cdk deploy (capture stdout + stderr)
  DEPLOY_OUTPUT=$(npx cdk deploy "$STACK_NAME" \
    --context mode=simple \
    --context workstation=true \
    --context allowedCidr="$ALLOWED_CIDR" \
    --context instanceType="$INSTANCE_TYPE" \
    --context availabilityZone="$AZ" \
    --require-approval never \
    2>&1) || DEPLOY_EXIT_CODE=$?

  # Check if deploy succeeded
  if [[ -z "${DEPLOY_EXIT_CODE:-}" ]]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo " SUCCESS: Workstation deployed in AZ $AZ"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo "$DEPLOY_OUTPUT"
    echo ""
    exit 0
  fi

  # Deploy failed — check if it's a capacity error
  echo ""
  echo "Deploy failed. Checking error type..."

  if is_capacity_error "$DEPLOY_OUTPUT"; then
    echo "    → Insufficient capacity in AZ $AZ"

    # Check if we have more AZs to try
    if [[ $ATTEMPT -lt ${#AZS[@]} ]]; then
      echo "    → Cleaning up failed stack and trying next AZ..."

      # BUG FIX 2: Only delete stack if it's in a known failed/rolled-back state
      # Do NOT delete stacks in progress or successfully created
      STACK_STATUS=$(get_stack_status)
      if [[ "$STACK_STATUS" == "DOES_NOT_EXIST" ]]; then
        echo "    → Stack does not exist. Proceeding to next AZ..."
      elif [[ "$STACK_STATUS" == "ROLLBACK_COMPLETE" || \
              "$STACK_STATUS" == "ROLLBACK_FAILED" || \
              "$STACK_STATUS" == "CREATE_FAILED" || \
              "$STACK_STATUS" == "UPDATE_ROLLBACK_COMPLETE" || \
              "$STACK_STATUS" == "UPDATE_ROLLBACK_FAILED" || \
              "$STACK_STATUS" == "DELETE_FAILED" ]]; then
        echo "    → Stack status: $STACK_STATUS. Deleting..."
        aws cloudformation delete-stack --stack-name "$STACK_NAME" --region "$REGION"
        echo "    → Waiting for stack deletion to complete..."
        aws cloudformation wait stack-delete-complete --stack-name "$STACK_NAME" --region "$REGION" || true
        echo "    → Stack deleted."
      else
        # Stack in unexpected state (in-progress or complete) — do not delete
        echo "    ERROR: Stack is in state $STACK_STATUS, expected a failed/rolled-back state."
        echo "           Will not delete stack in this state. Manual cleanup may be required."
        exit 1
      fi

      echo ""
      sleep 2  # Brief pause before next attempt
      continue
    else
      echo "    → No more AZs to try."
      echo ""
      echo "════════════════════════════════════════════════════════════════════════════════"
      echo " FAILURE: No capacity for $INSTANCE_TYPE in any AZ in $REGION"
      echo "════════════════════════════════════════════════════════════════════════════════"
      echo ""
      echo "Tried AZs: ${AZS[*]}"
      echo ""
      echo "Options:"
      echo "  1. Retry later (capacity fluctuates)"
      echo "  2. Try a different instance type (e.g., g6e.8xlarge, g5.2xlarge)"
      echo "  3. Try a different region"
      echo ""
      exit 1
    fi
  else
    # BUG FIX 3: Non-capacity error — log the full deploy output before stopping
    echo "    → Non-capacity error detected. Stopping."
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo " FAILURE: Deploy failed with non-capacity error"
    echo "════════════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "Deploy output:"
    echo "--------------------------------------------------------------------------------"
    echo "$DEPLOY_OUTPUT"
    echo "--------------------------------------------------------------------------------"
    echo ""
    exit 1
  fi
done

# Should never reach here, but just in case
echo "ERROR: Unexpectedly exhausted AZ list without success or failure."
exit 1
