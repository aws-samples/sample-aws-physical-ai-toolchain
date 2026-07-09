#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

# Uninstall AWS Prerequisites

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_DIR="$(dirname "$SCRIPT_DIR")"

# shellcheck source=../defaults.conf
source "$SETUP_DIR/defaults.conf"
# shellcheck source=../lib/common.sh
source "$SETUP_DIR/lib/common.sh"

show_help() {
  cat << EOF
Usage: $(basename "$0") [OPTIONS]

Uninstall AWS prerequisites (GPU Operator, LB Controller, External Secrets).

OPTIONS:
    -h, --help              Show this help message
    -t, --tf-dir DIR        Terraform directory
    --skip-gpu-operator     Skip GPU Operator uninstall
    --skip-lb-controller    Skip AWS LB Controller uninstall
    --skip-external-secrets Skip External Secrets uninstall
    --force                 Skip confirmation prompts

EXAMPLES:
    $(basename "$0")
    $(basename "$0") --skip-gpu-operator
EOF
}

# Default values
tf_dir="$SETUP_DIR/$DEFAULT_TF_DIR"
skip_gpu_operator=false
skip_lb_controller=false
skip_external_secrets=false
force=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)               show_help; exit 0 ;;
    -t|--tf-dir)             tf_dir="$2"; shift 2 ;;
    --skip-gpu-operator)     skip_gpu_operator=true; shift ;;
    --skip-lb-controller)    skip_lb_controller=true; shift ;;
    --skip-external-secrets) skip_external_secrets=true; shift ;;
    --force)                 force=true; shift ;;
    *)                       fatal "Unknown option: $1" ;;
  esac
done

main() {
  section "Uninstalling AWS Prerequisites"

  require_tools kubectl helm

  if [[ "$force" != "true" ]]; then
    warn "This will uninstall AWS prerequisites."
    warn "Ensure OSMO components are uninstalled first."
    read -rp "Continue? (y/N) " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || { info "Aborted."; exit 0; }
  fi

  # Read Terraform outputs
  local tf_outputs cluster_name region
  tf_outputs=$(read_terraform_outputs "$tf_dir")
  cluster_name=$(tf_require "$tf_outputs" "cluster_name" "EKS cluster name")
  region=$(tf_require "$tf_outputs" "aws_region" "AWS region")

  connect_eks "$region" "$cluster_name"

  # Uninstall GPU Operator
  if [[ "$skip_gpu_operator" != "true" ]]; then
    info "Uninstalling GPU Operator..."
    helm uninstall gpu-operator -n "$GPU_OPERATOR_NAMESPACE" 2>/dev/null || \
      warn "gpu-operator not found"
    kubectl delete namespace "$GPU_OPERATOR_NAMESPACE" --wait=false 2>/dev/null || true
  fi

  # Uninstall External Secrets
  if [[ "$skip_external_secrets" != "true" ]]; then
    info "Uninstalling External Secrets Operator..."
    kubectl delete clustersecretstore aws-secrets-manager 2>/dev/null || true
    helm uninstall external-secrets -n "$EXTERNAL_SECRETS_NAMESPACE" 2>/dev/null || \
      warn "external-secrets not found"
    kubectl delete namespace "$EXTERNAL_SECRETS_NAMESPACE" --wait=false 2>/dev/null || true
  fi

  # Uninstall AWS Load Balancer Controller
  if [[ "$skip_lb_controller" != "true" ]]; then
    info "Uninstalling AWS Load Balancer Controller..."
    helm uninstall aws-load-balancer-controller -n kube-system 2>/dev/null || \
      warn "aws-load-balancer-controller not found"
  fi

  # Delete storage classes
  info "Deleting custom storage classes..."
  kubectl delete storageclass gp3 gp3-high-perf 2>/dev/null || true

  pass "AWS Prerequisites uninstalled"
  echo
  info "To fully cleanup, run: terraform destroy in 001-iac/"
}

main "$@"
