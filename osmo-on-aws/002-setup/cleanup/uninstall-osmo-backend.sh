#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

# Uninstall OSMO Backend Operator

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

Uninstall OSMO Backend Operator.

OPTIONS:
    -h, --help              Show this help message
    -t, --tf-dir DIR        Terraform directory
    --delete-namespaces     Delete the operator and workflows namespaces
    --force                 Skip confirmation prompts

EXAMPLES:
    $(basename "$0")
    $(basename "$0") --delete-namespaces
EOF
}

# Default values
tf_dir="$SETUP_DIR/$DEFAULT_TF_DIR"
delete_namespaces=false
force=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)            show_help; exit 0 ;;
    -t|--tf-dir)          tf_dir="$2"; shift 2 ;;
    --delete-namespaces)  delete_namespaces=true; shift ;;
    --force)              force=true; shift ;;
    *)                    fatal "Unknown option: $1" ;;
  esac
done

main() {
  section "Uninstalling OSMO Backend Operator"

  require_tools kubectl helm

  if [[ "$force" != "true" ]]; then
    warn "This will uninstall the OSMO Backend Operator."
    warn "Running workloads in $OSMO_WORKFLOWS_NAMESPACE may be terminated."
    read -rp "Continue? (y/N) " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || { info "Aborted."; exit 0; }
  fi

  # Read Terraform outputs for cluster connection
  local tf_outputs cluster_name region
  tf_outputs=$(read_terraform_outputs "$tf_dir")
  cluster_name=$(tf_require "$tf_outputs" "cluster_name" "EKS cluster name")
  region=$(tf_require "$tf_outputs" "aws_region" "AWS region")

  connect_eks "$region" "$cluster_name"

  # Uninstall backend operator
  info "Uninstalling osmo-operator..."
  helm uninstall osmo-operator -n "$OSMO_OPERATOR_NAMESPACE" 2>/dev/null || \
    warn "osmo-operator not found or already uninstalled"

  # Delete backend operator token
  info "Deleting backend operator secrets..."
  kubectl delete secret osmo-operator-token -n "$OSMO_OPERATOR_NAMESPACE" 2>/dev/null || true

  # Delete namespaces if requested
  if [[ "$delete_namespaces" == "true" ]]; then
    warn "Deleting namespace: $OSMO_OPERATOR_NAMESPACE"
    kubectl delete namespace "$OSMO_OPERATOR_NAMESPACE" --wait=false 2>/dev/null || true
    warn "Deleting namespace: $OSMO_WORKFLOWS_NAMESPACE"
    kubectl delete namespace "$OSMO_WORKFLOWS_NAMESPACE" --wait=false 2>/dev/null || true
  fi

  pass "OSMO Backend Operator uninstalled"
}

main "$@"
