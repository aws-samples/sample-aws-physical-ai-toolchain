#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

# Deploy AWS Prerequisites: Load Balancer Controller, External Secrets, external-dns, Storage Classes

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=defaults.conf
source "$SCRIPT_DIR/defaults.conf"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

show_help() {
  cat << EOF
Usage: $(basename "$0") [OPTIONS]

Deploy AWS prerequisites for OSMO deployment.

OPTIONS:
    -h, --help              Show this help message
    -t, --tf-dir DIR        Terraform directory (default: $DEFAULT_TF_DIR)
    --skip-lb-controller    Skip AWS Load Balancer Controller
    --skip-external-secrets Skip External Secrets Operator
    --skip-external-dns     Skip external-dns controller
    --skip-storage-class    Skip storage class creation
    --config-preview        Print configuration and exit

EXAMPLES:
    $(basename "$0")
    $(basename "$0") --tf-dir ../001-iac
EOF
}

# Default values
tf_dir="$SCRIPT_DIR/$DEFAULT_TF_DIR"
skip_lb_controller=false
skip_external_secrets=false
skip_external_dns=false
skip_storage_class=false
skip_observability=false
config_preview=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)              show_help; exit 0 ;;
    -t|--tf-dir)            tf_dir="$2"; shift 2 ;;
    --skip-lb-controller)   skip_lb_controller=true; shift ;;
    --skip-external-secrets) skip_external_secrets=true; shift ;;
    --skip-external-dns)    skip_external_dns=true; shift ;;
    --skip-storage-class)   skip_storage_class=true; shift ;;
    --skip-observability)   skip_observability=true; shift ;;
    --config-preview)       config_preview=true; shift ;;
    *)                      fatal "Unknown option: $1" ;;
  esac
done

main() {
  section "Deploying AWS Prerequisites"

  require_tools aws kubectl helm jq

  # Read Terraform outputs
  info "Reading Terraform outputs from: $tf_dir"
  local tf_outputs
  tf_outputs=$(read_terraform_outputs "$tf_dir")

  # Extract required values
  local cluster_name region vpc_id
  cluster_name=$(tf_require "$tf_outputs" "cluster_name" "EKS cluster name")
  region=$(tf_require "$tf_outputs" "aws_region" "AWS region")
  vpc_id=$(tf_require "$tf_outputs" "vpc_id" "VPC ID")

  local lb_controller_role_arn external_secrets_role_arn external_dns_role_arn osmo_hostname
  lb_controller_role_arn=$(tf_require "$tf_outputs" "aws_lb_controller_role_arn" "LB Controller role ARN")
  external_secrets_role_arn=$(tf_require "$tf_outputs" "external_secrets_role_arn" "External Secrets role ARN")
  external_dns_role_arn=$(tf_require "$tf_outputs" "external_dns_role_arn" "external-dns role ARN")
  osmo_hostname=$(tf_require "$tf_outputs" "osmo_hostname" "OSMO hostname")

  # Print configuration
  section "Configuration"
  print_kv "Cluster" "$cluster_name"
  print_kv "Region" "$region"
  print_kv "VPC ID" "$vpc_id"
  print_kv "OSMO Hostname" "$osmo_hostname"
  print_kv "LB Controller Role" "$lb_controller_role_arn"
  print_kv "External Secrets Role" "$external_secrets_role_arn"
  print_kv "External DNS Role" "$external_dns_role_arn"

  if [[ "$config_preview" == "true" ]]; then
    info "Config preview mode - exiting"
    exit 0
  fi

  # Connect to cluster
  connect_eks "$region" "$cluster_name"

  # Deploy Storage Classes
  if [[ "$skip_storage_class" != "true" ]]; then
    section "Deploying Storage Classes"
    kubectl apply -f "$SCRIPT_DIR/manifests/storage-class.yaml"
    pass "Storage classes deployed"
  fi

  # Deploy AWS Load Balancer Controller
  if [[ "$skip_lb_controller" != "true" ]]; then
    section "Deploying AWS Load Balancer Controller"

    helm_repo_add eks "$AWS_HELM_REPO"

    helm_upgrade_install aws-load-balancer-controller eks/aws-load-balancer-controller \
      kube-system \
      --version "$AWS_LB_CONTROLLER_VERSION" \
      -f "$SCRIPT_DIR/values/aws-load-balancer-controller.yaml" \
      --set clusterName="$cluster_name" \
      --set region="$region" \
      --set vpcId="$vpc_id" \
      --set serviceAccount.annotations."eks\.amazonaws\.com/role-arn"="$lb_controller_role_arn" \
      --timeout "$HELM_TIMEOUT"

    wait_for_deployment aws-load-balancer-controller kube-system
    pass "AWS Load Balancer Controller deployed"
  fi

  # Deploy External Secrets Operator
  if [[ "$skip_external_secrets" != "true" ]]; then
    section "Deploying External Secrets Operator"

    helm_repo_add external-secrets "$EXTERNAL_SECRETS_REPO"

    helm_upgrade_install external-secrets external-secrets/external-secrets \
      "$EXTERNAL_SECRETS_NAMESPACE" \
      --version "$EXTERNAL_SECRETS_VERSION" \
      --set serviceAccount.annotations."eks\.amazonaws\.com/role-arn"="$external_secrets_role_arn" \
      --set installCRDs=true \
      --timeout "$HELM_TIMEOUT"

    wait_for_deployment external-secrets "$EXTERNAL_SECRETS_NAMESPACE"

    # Wait for CRDs to be established before creating ClusterSecretStore
    info "Waiting for External Secrets CRDs to be established..."
    kubectl wait --for=condition=established crd/clustersecretstores.external-secrets.io --timeout="${KUBECTL_TIMEOUT:-120s}" || \
      fatal "ClusterSecretStore CRD not ready in time"
    # Brief pause so the API server can serve the CRD
    sleep 3

    # Create ClusterSecretStore for AWS Secrets Manager
    info "Creating ClusterSecretStore for AWS Secrets Manager"
    kubectl apply -f - << EOF
apiVersion: external-secrets.io/v1beta1
kind: ClusterSecretStore
metadata:
  name: aws-secrets-manager
spec:
  provider:
    aws:
      service: SecretsManager
      region: $region
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets
            namespace: $EXTERNAL_SECRETS_NAMESPACE
EOF
    pass "External Secrets Operator deployed"
  fi

  # Deploy external-dns
  if [[ "$skip_external_dns" != "true" ]]; then
    section "Deploying external-dns"

    helm_repo_add external-dns "$EXTERNAL_DNS_REPO"

    # Extract zone name from osmo_hostname (strip the first label)
    local zone_name="${osmo_hostname#*.}"

    # external-dns TXT-registry owner id; override to isolate a second deployment sharing a parent zone
    local txt_owner_id="${EXTERNAL_DNS_TXT_OWNER_ID:-osmo}"

    helm_upgrade_install external-dns external-dns/external-dns \
      kube-system \
      --version "$EXTERNAL_DNS_VERSION" \
      --set provider.name=aws \
      --set policy=sync \
      --set registry=txt \
      --set txtOwnerId="$txt_owner_id" \
      --set "domainFilters[0]=$zone_name" \
      --set "sources[0]=ingress" \
      --set serviceAccount.annotations."eks\.amazonaws\.com/role-arn"="$external_dns_role_arn" \
      --timeout "$HELM_TIMEOUT"

    wait_for_deployment external-dns kube-system
    pass "external-dns deployed"
  fi

  # Deploy observability: Fluent Bit log shipper + AMP scraper RBAC
  if [[ "$skip_observability" != "true" ]]; then
    # #1 Logs — aws-for-fluent-bit DaemonSet -> CloudWatch Logs (when Terraform
    # provisioned the Fluent Bit IRSA role + log group).
    local fluentbit_role_arn log_group_name
    fluentbit_role_arn=$(tf_get "$tf_outputs" "fluentbit_role_arn" "")
    log_group_name=$(tf_get "$tf_outputs" "cloudwatch_log_group_name" "")

    if [[ -n "$fluentbit_role_arn" && "$fluentbit_role_arn" != "null" ]]; then
      section "Deploying Fluent Bit (CloudWatch Logs)"
      ensure_namespace amazon-cloudwatch
      helm_repo_add eks "$AWS_HELM_REPO"
      helm_upgrade_install aws-for-fluent-bit eks/aws-for-fluent-bit \
        amazon-cloudwatch \
        --set serviceAccount.create=true \
        --set serviceAccount.name=aws-for-fluent-bit \
        --set serviceAccount.annotations."eks\.amazonaws\.com/role-arn"="$fluentbit_role_arn" \
        --set cloudWatchLogs.enabled=true \
        --set cloudWatchLogs.region="$region" \
        --set cloudWatchLogs.logGroupName="$log_group_name" \
        --set cloudWatchLogs.logStreamPrefix="${cluster_name}-" \
        --set cloudWatchLogs.autoCreateGroup=false \
        --set firehose.enabled=false \
        --set kinesis.enabled=false \
        --set elasticsearch.enabled=false \
        --timeout "$HELM_TIMEOUT"
      pass "Fluent Bit deployed -> $log_group_name"
    else
      info "CloudWatch logging not provisioned (enable_cloudwatch_logging=false) — skipping Fluent Bit"
    fi

    # #2 Metrics — the AWS-managed AMP scraper needs NO in-cluster RBAC here.
    # This cluster uses EKS access entries, so AMP auto-creates the scraper's
    # access entry and associates the AmazonPrometheusScraperPolicy cluster
    # access policy automatically (manual ClusterRole/Binding is only needed
    # for the legacy aws-auth ConfigMap path).
    local amp_workspace_id
    amp_workspace_id=$(tf_get "$tf_outputs" "amp_workspace_id" "")
    if [[ -n "$amp_workspace_id" && "$amp_workspace_id" != "null" ]]; then
      info "AMP workspace $amp_workspace_id — scraper cluster access is managed by AMP (no RBAC to apply)"
    fi
  fi

  section "AWS Prerequisites Deployment Complete"
  info "Next: Run 02-deploy-gpu-infrastructure.sh (if deploying GPU workloads)"
}

main "$@"
