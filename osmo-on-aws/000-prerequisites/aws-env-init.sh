#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

# Initialize AWS environment and export Terraform variables

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

info() { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*" >&2; }
error() { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; }
fatal() { error "$@"; exit 1; }

show_help() {
  cat << EOF
Usage: $(basename "$0") [OPTIONS]

Initialize AWS environment for OSMO deployment.

OPTIONS:
    -h, --help              Show this help message
    -r, --region REGION     AWS region (default: us-west-2)
    -p, --profile PROFILE   AWS profile to use
    --export                Export environment variables to file

EXAMPLES:
    $(basename "$0") --region us-west-2
    $(basename "$0") --profile my-profile --export
EOF
}

# Default values
AWS_REGION="${AWS_REGION:-us-west-2}"
AWS_PROFILE="${AWS_PROFILE:-}"
EXPORT_FILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)      show_help; exit 0 ;;
    -r|--region)    AWS_REGION="$2"; shift 2 ;;
    -p|--profile)   AWS_PROFILE="$2"; shift 2 ;;
    --export)       EXPORT_FILE="${SCRIPT_DIR}/.env"; shift ;;
    *)              fatal "Unknown option: $1" ;;
  esac
done

main() {
  info "Initializing AWS environment..."

  # Set AWS profile if specified
  if [[ -n "$AWS_PROFILE" ]]; then
    export AWS_PROFILE
    info "Using AWS profile: $AWS_PROFILE"
  fi

  # Verify AWS credentials
  info "Verifying AWS credentials..."
  if ! aws sts get-caller-identity &>/dev/null; then
    fatal "AWS credentials not configured. Run 'aws configure' first."
  fi

  local account_id
  account_id=$(aws sts get-caller-identity --query Account --output text)
  local identity_arn
  identity_arn=$(aws sts get-caller-identity --query Arn --output text)

  info "AWS Account: $account_id"
  info "Identity: $identity_arn"
  info "Region: $AWS_REGION"

  # Export region for the AWS CLI
  export AWS_REGION="$AWS_REGION"
  export AWS_DEFAULT_REGION="$AWS_REGION"

  # Export to file if requested
  if [[ -n "$EXPORT_FILE" ]]; then
    cat > "$EXPORT_FILE" << EOF
# OSMO on AWS Environment Variables
# Generated on $(date)
# Source this file: source $EXPORT_FILE

export AWS_REGION="$AWS_REGION"
export AWS_DEFAULT_REGION="$AWS_REGION"
EOF
    [[ -n "$AWS_PROFILE" ]] && echo "export AWS_PROFILE=\"$AWS_PROFILE\"" >> "$EXPORT_FILE"
    info "Environment variables exported to: $EXPORT_FILE"
    info "Run: source $EXPORT_FILE"
  fi

  echo
  info "Environment initialized successfully!"
}

main
