#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

# Uninstall OSMO Control Plane

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETUP_DIR="$(dirname "$SCRIPT_DIR")"

# shellcheck source=../lib/common.sh
source "$SETUP_DIR/lib/common.sh"
# shellcheck source=../defaults.conf
source "$SETUP_DIR/defaults.conf"

show_help() {
  cat << EOF
Usage: $(basename "$0") [OPTIONS]

Uninstall OSMO Control Plane components.

OPTIONS:
    -h, --help              Show this help message
    -t, --tf-dir DIR        Terraform directory
    --delete-namespace      Delete the OSMO namespace
    --delete-secrets        Delete all secrets
    --force                 Skip confirmation prompts

EXAMPLES:
    $(basename "$0")
    $(basename "$0") --delete-namespace --delete-secrets
EOF
}

# Default values
tf_dir="$SETUP_DIR/$DEFAULT_TF_DIR"
delete_namespace=false
delete_secrets=false
force=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)           show_help; exit 0 ;;
    -t|--tf-dir)         tf_dir="$2"; shift 2 ;;
    --delete-namespace)  delete_namespace=true; shift ;;
    --delete-secrets)    delete_secrets=true; shift ;;
    --force)             force=true; shift ;;
    *)                   fatal "Unknown option: $1" ;;
  esac
done

main() {
  section "Uninstalling OSMO Control Plane"

  require_tools kubectl helm

  if [[ "$force" != "true" ]]; then
    warn "This will uninstall the OSMO Control Plane."
    warn "All OSMO services will become unavailable."
    read -rp "Continue? (y/N) " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || { info "Aborted."; exit 0; }
  fi

  # Read Terraform outputs
  local tf_outputs cluster_name region
  tf_outputs=$(read_terraform_outputs "$tf_dir")
  cluster_name=$(tf_require "$tf_outputs" "cluster_name" "EKS cluster name")
  region=$(tf_require "$tf_outputs" "aws_region" "AWS region")

  connect_eks "$region" "$cluster_name"

  # 6.3: router + UI are part of the single `service` release. Uninstall the
  # legacy standalone releases too (harmless no-ops on a fresh 6.3 install).
  info "Uninstalling service (includes router + UI in 6.3)..."
  helm uninstall service -n "$OSMO_NAMESPACE" 2>/dev/null || \
    warn "service not found"

  for legacy in ui router; do
    if helm status "$legacy" -n "$OSMO_NAMESPACE" &>/dev/null; then
      info "Uninstalling legacy $legacy release..."
      helm uninstall "$legacy" -n "$OSMO_NAMESPACE" 2>/dev/null || true
    fi
  done

  # Delete secrets if requested
  if [[ "$delete_secrets" == "true" ]]; then
    warn "Deleting secrets..."
    kubectl delete secret ngc-registry-secret db-secret redis-secret \
      oauth2-proxy-secrets vault-secrets \
      -n "$OSMO_NAMESPACE" 2>/dev/null || true
    kubectl delete configmap mek-config \
      -n "$OSMO_NAMESPACE" 2>/dev/null || true
  fi

  # Delete namespace if requested
  if [[ "$delete_namespace" == "true" ]]; then
    warn "Deleting namespace: $OSMO_NAMESPACE"
    kubectl delete namespace "$OSMO_NAMESPACE" --wait=false 2>/dev/null || true
  fi

  pass "OSMO Control Plane uninstalled"
}

main "$@"
