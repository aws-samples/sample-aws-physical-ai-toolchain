#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

# Verify all prerequisites for OSMO on AWS deployment

set -o errexit
set -o nounset
set -o pipefail

info() { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
pass() { printf '\033[1;32m[PASS]\033[0m  %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*" >&2; }
fail() { printf '\033[1;31m[FAIL]\033[0m  %s\n' "$*" >&2; }

ERRORS=0

check_command() {
  local cmd="$1"

  if ! command -v "$cmd" &>/dev/null; then
    fail "$cmd is not installed"
    ((ERRORS++))
    return 1
  fi

  local version
  case "$cmd" in
    aws)        version=$(aws --version 2>&1 | cut -d/ -f2 | cut -d' ' -f1) ;;
    kubectl)    version=$(kubectl version --client -o json 2>/dev/null | jq -r '.clientVersion.gitVersion' | tr -d 'v') ;;
    helm)       version=$(helm version --short | tr -d 'v' | cut -d'+' -f1) ;;
    terraform)  version=$(terraform version -json | jq -r '.terraform_version') ;;
    jq)         version=$(jq --version | tr -d 'jq-') ;;
    pre-commit) version=$(pre-commit --version | awk '{print $2}') ;;
    checkov)    version=$(checkov --version 2>/dev/null | sed -n '1p') ;;
    tflint)     version=$(tflint --version 2>/dev/null | awk 'NR==1{print $3}') ;;
    shellcheck) version=$(shellcheck --version | grep version: | awk '{print $2}') ;;
    gitleaks)   version=$(gitleaks version 2>/dev/null || echo "unknown") ;;
    yq)         version=$(yq --version 2>/dev/null | awk '{print $NF}' | sed 's/^v//') ;;
    openssl)    version=$(openssl version 2>/dev/null | awk '{print $2}') ;;
    *)          version="unknown" ;;
  esac

  pass "$cmd installed (v${version})"
  return 0
}

check_optional_command() {
  local cmd="$1"

  if ! command -v "$cmd" &>/dev/null; then
    warn "$cmd is not installed (optional)"
    return 0
  fi

  local version
  case "$cmd" in
    pre-commit) version=$(pre-commit --version | awk '{print $2}') ;;
    checkov)    version=$(checkov --version 2>/dev/null | sed -n '1p') ;;
    tflint)     version=$(tflint --version 2>/dev/null | awk 'NR==1{print $3}') ;;
    shellcheck) version=$(shellcheck --version | grep version: | awk '{print $2}') ;;
    gitleaks)   version=$(gitleaks version 2>/dev/null || echo "unknown") ;;
    *)          version="unknown" ;;
  esac

  pass "$cmd installed (v$version)"
  return 0
}

check_aws_credentials() {
  info "Checking AWS credentials..."

  if ! aws sts get-caller-identity &>/dev/null; then
    fail "AWS credentials not configured or invalid"
    ((ERRORS++))
    return 1
  fi

  local account_id identity
  account_id=$(aws sts get-caller-identity --query Account --output text)
  identity=$(aws sts get-caller-identity --query Arn --output text)

  pass "AWS credentials valid"
  info "  Account: $account_id"
  info "  Identity: $identity"
  return 0
}

check_aws_permissions() {
  info "Checking AWS permissions..."

  # Check basic permissions needed for deployment
  local services=("ec2" "eks" "rds" "elasticache" "s3" "secretsmanager" "iam")
  local missing=()

  for svc in "${services[@]}"; do
    case "$svc" in
      ec2)
        aws ec2 describe-vpcs --max-items 1 &>/dev/null || missing+=("$svc")
        ;;
      eks)
        aws eks list-clusters --max-items 1 &>/dev/null || missing+=("$svc")
        ;;
      s3)
        aws s3 ls &>/dev/null || missing+=("$svc")
        ;;
      secretsmanager)
        aws secretsmanager list-secrets --max-results 1 &>/dev/null || missing+=("$svc")
        ;;
      iam)
        aws iam get-user &>/dev/null 2>&1 || aws iam list-roles --max-items 1 &>/dev/null || missing+=("$svc")
        ;;
    esac
  done

  if [[ ${#missing[@]} -gt 0 ]]; then
    warn "Limited permissions for: ${missing[*]}"
    warn "Some operations may require additional IAM permissions"
  else
    pass "Basic AWS permissions verified"
  fi
}

main() {
  echo "======================================"
  echo "OSMO on AWS Prerequisites Verification"
  echo "======================================"
  echo

  info "Checking required tools..."
  check_command aws
  check_command kubectl
  check_command helm
  check_command terraform
  check_command jq
  check_command yq
  check_command openssl
  echo

  info "Checking security & development tools (optional)..."
  check_optional_command pre-commit
  check_optional_command checkov
  check_optional_command tflint
  check_optional_command shellcheck
  check_optional_command gitleaks
  echo

  check_aws_credentials
  echo

  check_aws_permissions
  echo

  echo "======================================"
  if [[ $ERRORS -gt 0 ]]; then
    fail "Verification failed with $ERRORS error(s)"
    echo
    info "Please install missing tools and try again."
    info "Run: ./install-tools.sh"
    exit 1
  else
    pass "All prerequisites verified!"
    echo
    info "Ready to deploy. Next steps:"
    info "  1. cd ../001-iac"
    info "  2. cp terraform.tfvars.example terraform.tfvars"
    info "  3. Edit terraform.tfvars"
    info "  4. terraform init && terraform apply"
  fi
}

main
