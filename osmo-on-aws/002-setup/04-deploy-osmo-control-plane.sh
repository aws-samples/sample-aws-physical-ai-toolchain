#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

# Deploy OSMO Control Plane (6.3 consolidated `service` chart: API + Router + UI
# + Gateway) with Keycloak IdP and ConfigMap-based configuration.

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck source=defaults.conf
source "$SCRIPT_DIR/defaults.conf"

show_help() {
  cat << EOF
Usage: $(basename "$0") [OPTIONS]

Deploy OSMO Control Plane components (Service, Router, Web UI) with Keycloak IdP.

OPTIONS:
    -h, --help              Show this help message
    -t, --tf-dir DIR        Terraform directory (default: $DEFAULT_TF_DIR)
    --ngc-api-key KEY       NGC API key (or set NGC_API_KEY env var)
    --skip-secrets          Skip secret creation (use existing)
    --skip-mek              Skip MEK generation (use existing)
    --config-preview        Print configuration and exit

EXAMPLES:
    $(basename "$0") --ngc-api-key "your-key"
    $(basename "$0") --skip-secrets
EOF
}

# Default values
tf_dir="$SCRIPT_DIR/$DEFAULT_TF_DIR"
ngc_api_key="${NGC_API_KEY:-}"
skip_secrets=false
skip_mek=false
config_preview=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)          show_help; exit 0 ;;
    -t|--tf-dir)        tf_dir="$2"; shift 2 ;;
    --ngc-api-key)      ngc_api_key="$2"; shift 2 ;;
    --skip-secrets)     skip_secrets=true; shift ;;
    --skip-mek)         skip_mek=true; shift ;;
    --config-preview)   config_preview=true; shift ;;
    *)                  fatal "Unknown option: $1" ;;
  esac
done

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------

main() {
  section "Deploying OSMO Control Plane (6.3 + Keycloak)"

  require_tools aws kubectl helm jq openssl yq

  # Read Terraform outputs
  info "Reading Terraform outputs from: $tf_dir"
  local tf_outputs
  tf_outputs=$(read_terraform_outputs "$tf_dir")

  # Extract required values
  local cluster_name region deployment_mode osmo_hostname
  cluster_name=$(tf_require "$tf_outputs" "cluster_name" "EKS cluster name")
  region=$(tf_require "$tf_outputs" "aws_region" "AWS region")
  deployment_mode=$(tf_get "$tf_outputs" "deployment_mode" "full")
  osmo_hostname=$(tf_require "$tf_outputs" "osmo_hostname" "OSMO hostname")

  if ! should_deploy_control_plane "$deployment_mode"; then
    info "Deployment mode is '$deployment_mode' - skipping Control Plane"
    exit 0
  fi

  # Keycloak configuration from Terraform outputs
  local kc_issuer_url kc_jwks_uri kc_token_endpoint kc_authorize_endpoint
  local kc_device_endpoint kc_logout_endpoint kc_browser_client_id kc_device_client_id
  local kc_auth_hostname
  kc_issuer_url=$(tf_require "$tf_outputs" "keycloak_issuer_url" "Keycloak issuer URL")
  kc_jwks_uri=$(tf_require "$tf_outputs" "keycloak_jwks_uri" "Keycloak JWKS URI")
  kc_token_endpoint=$(tf_require "$tf_outputs" "keycloak_token_endpoint" "Keycloak token endpoint")
  kc_authorize_endpoint=$(tf_require "$tf_outputs" "keycloak_authorize_endpoint" "Keycloak authorize endpoint")
  kc_device_endpoint=$(tf_require "$tf_outputs" "keycloak_device_endpoint" "Keycloak device endpoint")
  kc_logout_endpoint=$(tf_require "$tf_outputs" "keycloak_logout_endpoint" "Keycloak logout endpoint")
  kc_browser_client_id=$(tf_require "$tf_outputs" "keycloak_browser_client_id" "Keycloak browser client ID")
  kc_device_client_id=$(tf_require "$tf_outputs" "keycloak_device_client_id" "Keycloak device client ID")
  kc_auth_hostname=$(tf_require "$tf_outputs" "osmo_auth_hostname" "OSMO auth hostname")

  # Database configuration
  local rds_endpoint rds_port rds_database rds_username rds_password_secret_arn
  rds_endpoint=$(tf_require "$tf_outputs" "rds_endpoint" "RDS endpoint")
  rds_port=$(tf_get "$tf_outputs" "rds_port" "5432")
  rds_database=$(tf_get "$tf_outputs" "rds_database_name" "osmo")
  rds_username=$(tf_get "$tf_outputs" "rds_username" "osmo_admin")
  rds_password_secret_arn=$(tf_require "$tf_outputs" "rds_password_secret_arn" "RDS password secret ARN")

  # Redis configuration
  local redis_endpoint redis_port redis_auth_token_secret_arn
  redis_endpoint=$(tf_require "$tf_outputs" "redis_endpoint" "Redis endpoint")
  redis_port=$(tf_get "$tf_outputs" "redis_port" "6379")
  redis_auth_token_secret_arn=$(tf_require "$tf_outputs" "redis_auth_token_secret_arn" "Redis auth token secret ARN")

  # S3 configuration
  local s3_workflows_bucket s3_datasets_bucket
  s3_workflows_bucket=$(tf_require "$tf_outputs" "s3_workflows_bucket_name" "S3 workflows bucket")
  s3_datasets_bucket=$(tf_require "$tf_outputs" "s3_datasets_bucket_name" "S3 datasets bucket")

  # IRSA role
  local osmo_service_role_arn
  osmo_service_role_arn=$(tf_require "$tf_outputs" "osmo_service_role_arn" "OSMO service role ARN")

  # ACM certificate for OSMO service
  local acm_certificate_arn
  acm_certificate_arn=$(tf_require "$tf_outputs" "acm_certificate_arn" "ACM certificate ARN")

  # ALB security group (optional)
  local alb_security_group_id
  alb_security_group_id=$(tf_get "$tf_outputs" "alb_security_group_id" "")

  # Derived values
  local cookie_domain=".${osmo_hostname}"

  # Print configuration
  section "Configuration"
  print_kv "Cluster" "$cluster_name"
  print_kv "Region" "$region"
  print_kv "OSMO Version" "$OSMO_VERSION"
  print_kv "Chart Version" "$OSMO_CHART_VERSION"
  print_kv "OSMO Hostname" "$osmo_hostname"
  print_kv "IdP" "Keycloak (${kc_auth_hostname})"
  print_kv "KC Issuer" "$kc_issuer_url"
  print_kv "KC Browser Client" "$kc_browser_client_id"
  print_kv "KC Device Client" "$kc_device_client_id"
  print_kv "KC JWKS URI" "$kc_jwks_uri"
  print_kv "RDS Endpoint" "$rds_endpoint"
  print_kv "Redis Endpoint" "$redis_endpoint"
  print_kv "S3 Workflows" "$s3_workflows_bucket"
  print_kv "S3 Datasets" "$s3_datasets_bucket"

  if [[ "$config_preview" == "true" ]]; then
    info "Config preview mode - exiting"
    exit 0
  fi

  # Connect to cluster
  connect_eks "$region" "$cluster_name"
  ensure_namespace "$OSMO_NAMESPACE"

  local rds_host="${rds_endpoint%:*}"

  #----------------------------------------------------------------------------
  # Fetch Helm Charts (pre-release RC from NGC)
  #----------------------------------------------------------------------------
  section "Fetching Helm Charts"

  # 6.3: router and web-ui are folded into the single `service` chart.
  local service_chart
  service_chart=$(fetch_osmo_chart "service")

  pass "Chart fetched"

  #----------------------------------------------------------------------------
  # Create Secrets
  #----------------------------------------------------------------------------
  if [[ "$skip_secrets" != "true" ]]; then
    section "Creating Kubernetes Secrets"

    # NGC Registry Secret
    if [[ -n "$ngc_api_key" ]]; then
      info "Creating NGC registry secret"
      # shellcheck disable=SC2016
      kubectl create secret docker-registry ngc-registry-secret \
        --namespace "$OSMO_NAMESPACE" \
        --docker-server=nvcr.io \
        --docker-username='$oauthtoken' \
        --docker-password="$ngc_api_key" \
        --dry-run=client -o yaml | kubectl apply -f -
    else
      warn "NGC API key not provided - using existing secret or pulling from public registry"
    fi

    # Database password
    info "Creating database password secret"
    local db_password
    db_password=$(get_secret_value "$rds_password_secret_arn" "password" "$region")
    kubectl create secret generic db-secret \
      --namespace "$OSMO_NAMESPACE" \
      --from-literal=db-password="$db_password" \
      --dry-run=client -o yaml | kubectl apply -f -

    # Redis auth token
    info "Creating Redis auth token secret"
    local redis_auth_token
    redis_auth_token=$(get_secret_value "$redis_auth_token_secret_arn" "auth_token" "$region")
    kubectl create secret generic redis-secret \
      --namespace "$OSMO_NAMESPACE" \
      --from-literal=redis-password="$redis_auth_token" \
      --dry-run=client -o yaml | kubectl apply -f -

    # OAuth2 Proxy secrets — the browser client secret is stored by 03-deploy-keycloak.sh.
    # If oauth2-proxy-secrets already exists (from the Keycloak deploy step), skip.
    if kubectl get secret oauth2-proxy-secrets -n "$OSMO_NAMESPACE" &>/dev/null; then
      info "oauth2-proxy-secrets already exists (created by 03-deploy-keycloak.sh)"
    else
      warn "oauth2-proxy-secrets not found — run 03-deploy-keycloak.sh first or create it manually"
    fi

    pass "Secrets created"
  fi

  #----------------------------------------------------------------------------
  # Generate MEK
  #----------------------------------------------------------------------------
  if [[ "$skip_mek" != "true" ]]; then
    section "Generating Master Encryption Key"
    # 6.3: the service chart mounts the mek-config ConfigMap (key mek.yaml)
    # directly into the services, so only the ConfigMap is required — the
    # 6.2 vault-secrets Secret + manual deployment volume patch are obsolete.
    generate_mek_config "mek-config" "$OSMO_NAMESPACE" | kubectl apply -f -
    pass "MEK generated (mek-config ConfigMap)"
  fi

  #----------------------------------------------------------------------------
  # Generate dynamic ConfigMap-mode overlay
  #
  # ConfigMap mode (services.configs.*) is the source of truth for config.
  # Static config lives in values/osmo-control-plane.yaml; the dynamic pieces
  # (dataset bucket path, workflow scratch/log storage URLs, and the image
  # registry validation skip-list) are rendered here and layered via -f.
  #
  # Credential model (per the 6.3 docs):
  #   - Internal workflow_data / workflow_log buckets  -> STATIC IAM-user keys
  #     (required by the agent; consumed by osmo-ctrl which needs access_key_id).
  #   - User dataset buckets (task inputs/outputs)      -> pod IRSA (osmo-workflow
  #     ServiceAccount); registered path-only, no static credential.
  #----------------------------------------------------------------------------
  section "Generating ConfigMap overlay"

  local account_id ecr_registry
  account_id="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
  ecr_registry="${account_id}.dkr.ecr.${region}.amazonaws.com"

  # Internal workflow_data / workflow_log buckets use STATIC credentials (the
  # dedicated IAM user's S3 keys from Secrets Manager). The agent requires
  # workflow_data.credential to be set and treats data endpoints as
  # StaticDataCredential; osmo-ctrl in the task pod resolves them via the local
  # config (needs access_key_id). IRSA / pod workload identity applies to the
  # USER DATA (dataset) buckets only — see the 6.3 "Workload Identity for
  # Workflow Pods" guide. The dataset bucket below is therefore path-only and
  # osmo-ctrl reaches it via the osmo-workflow IRSA ServiceAccount.
  local s3_creds_arn s3_creds_json s3_access_key_id s3_secret_access_key
  s3_creds_arn=$(tf_require "$tf_outputs" "osmo_s3_credentials_secret_arn" "S3 credentials secret")
  s3_creds_json=$(aws secretsmanager get-secret-value \
    --secret-id "$s3_creds_arn" --region "$region" --query SecretString --output text)
  s3_access_key_id=$(echo "$s3_creds_json" | jq -r '.access_key_id')
  s3_secret_access_key=$(echo "$s3_creds_json" | jq -r '.secret_access_key')
  if [[ -z "$s3_access_key_id" || "$s3_access_key_id" == "null" ]]; then
    fatal "Failed to retrieve S3 access key from Secrets Manager ($s3_creds_arn)"
  fi
  info "S3 storage credentials retrieved (access key: ${s3_access_key_id:0:4}...)"

  local config_out_dir="$SCRIPT_DIR/config/out"
  mkdir -p "$config_out_dir"
  local configs_overlay="$config_out_dir/configs-overlay.yaml"

  cat > "$configs_overlay" <<EOF
services:
  configs:
    # Service base URL — osmo-ctrl in task pods derives the logger / token
    # refresh / router websocket endpoints from this. If unset, osmo-ctrl
    # builds "ws://:80/..." (empty host) and cannot stream logs or refresh its
    # JWT. The chart only auto-derives this when configs.service is non-empty.
    service:
      service_base_url: "https://${osmo_hostname}"
    workflow:
      # OSMO-injected containers on every workflow task pod (the osmo-ctrl
      # sidecar + init container). Without these the backend operator builds a
      # pod with empty container images and K8s rejects it (422 Invalid:
      # spec.containers[1].image / initContainers[0].image Required).
      backend_images:
        init: "${OSMO_IMAGE_LOCATION}/init-container:${OSMO_VERSION}"
        client: "${OSMO_IMAGE_LOCATION}/client:${OSMO_VERSION}"
      # Internal workflow scratch storage. Static credential is REQUIRED by the
      # agent (raises "Workflow data credential is not set" otherwise) and is
      # consumed by osmo-ctrl, which needs access_key_id in its local config.
      workflow_data:
        base_url: "s3://${s3_workflows_bucket}"
        credential:
          endpoint: "s3://${s3_workflows_bucket}"
          region: ${region}
          access_key_id: "${s3_access_key_id}"
          access_key: "${s3_secret_access_key}"
      # Internal workflow log storage. LogConfig has no base_url, so the bucket
      # location is carried by the credential endpoint (static credential).
      workflow_log:
        credential:
          endpoint: "s3://${s3_workflows_bucket}/osmo_workflow_log"
          region: ${region}
          access_key_id: "${s3_access_key_id}"
          access_key: "${s3_secret_access_key}"
      credential_config:
        disable_registry_validation:
          - nvcr.io
          - ${ecr_registry}
        disable_data_validation:
          - s3
    # USER DATA (dataset) bucket — path-only, no credential. osmo-ctrl reaches
    # it via the osmo-workflow IRSA ServiceAccount (pod workload identity).
    dataset:
      default_bucket: default
      buckets:
        default:
          dataset_path: "s3://${s3_datasets_bucket}/osmo-datasets"
          region: ${region}
EOF
  pass "ConfigMap overlay generated: $configs_overlay"

  #----------------------------------------------------------------------------
  # Helper: build --set args for ALB security group (gateway ingress)
  #----------------------------------------------------------------------------
  build_sg_args() {
    local prefix="${1:?ingress path prefix required}"
    local args=()
    if [[ -n "$alb_security_group_id" ]]; then
      args+=(--set "${prefix}.alb\\.ingress\\.kubernetes\\.io/security-groups=$alb_security_group_id")
      args+=(--set-string "${prefix}.alb\\.ingress\\.kubernetes\\.io/manage-backend-security-group-rules=true")
    fi
    echo "${args[@]}"
  }

  #----------------------------------------------------------------------------
  # Deploy the consolidated OSMO service chart (API + router + UI + gateway)
  #----------------------------------------------------------------------------
  section "Deploying OSMO Service (API + Router + UI + Gateway)"

  local gw_sg_args
  gw_sg_args=$(build_sg_args "gateway.envoy.ingress.annotations")

  info "Installing/upgrading Helm release: service"
  # shellcheck disable=SC2086
  helm upgrade --install service "$service_chart" \
    --namespace "$OSMO_NAMESPACE" \
    --create-namespace \
    -f "$SCRIPT_DIR/values/osmo-control-plane.yaml" \
    -f "$configs_overlay" \
    --set global.osmoImageTag="$OSMO_VERSION" \
    --set global.hostname="$osmo_hostname" \
    --set global.serviceAccountName=osmo-service \
    --set serviceAccount.create=true \
    --set serviceAccount.name=osmo-service \
    --set serviceAccount.annotations."eks\.amazonaws\.com/role-arn"="$osmo_service_role_arn" \
    --set services.postgres.serviceName="$rds_host" \
    --set services.postgres.port="$rds_port" \
    --set services.postgres.db="$rds_database" \
    --set services.postgres.user="$rds_username" \
    --set services.redis.serviceName="$redis_endpoint" \
    --set services.redis.port="$redis_port" \
    --set services.service.hostname="$osmo_hostname" \
    --set services.service.auth.enabled=true \
    --set "services.service.auth.browser_endpoint=$kc_authorize_endpoint" \
    --set "services.service.auth.browser_client_id=$kc_browser_client_id" \
    --set "services.service.auth.token_endpoint=$kc_token_endpoint" \
    --set "services.service.auth.logout_endpoint=$kc_logout_endpoint" \
    --set "services.service.auth.device_endpoint=$kc_device_endpoint" \
    --set "services.service.auth.device_client_id=$kc_device_client_id" \
    --set gateway.envoy.hostname="$osmo_hostname" \
    --set "gateway.envoy.idp.host=$kc_auth_hostname" \
    --set "gateway.envoy.ingress.albAnnotations.sslCertArn=$acm_certificate_arn" \
    $gw_sg_args \
    --set "gateway.envoy.jwt.providers[0].issuer=$kc_issuer_url" \
    --set "gateway.envoy.jwt.providers[0].jwks_uri=$kc_jwks_uri" \
    --set "gateway.envoy.jwt.providers[0].audience=$kc_browser_client_id" \
    --set "gateway.envoy.jwt.providers[0].user_claim=preferred_username" \
    --set "gateway.envoy.jwt.providers[0].cluster=idp" \
    --set "gateway.envoy.jwt.providers[1].issuer=$kc_issuer_url" \
    --set "gateway.envoy.jwt.providers[1].jwks_uri=$kc_jwks_uri" \
    --set "gateway.envoy.jwt.providers[1].audience=$kc_device_client_id" \
    --set "gateway.envoy.jwt.providers[1].user_claim=preferred_username" \
    --set "gateway.envoy.jwt.providers[1].cluster=idp" \
    --set "gateway.envoy.jwt.providers[2].issuer=osmo" \
    --set "gateway.envoy.jwt.providers[2].audience=osmo" \
    --set "gateway.envoy.jwt.providers[2].jwks_uri=https://osmo-service/api/auth/keys" \
    --set "gateway.envoy.jwt.providers[2].user_claim=unique_name" \
    --set "gateway.envoy.jwt.providers[2].cluster=osmo-service-jwks" \
    --set "gateway.oauth2Proxy.oidcIssuerUrl=$kc_issuer_url" \
    --set "gateway.oauth2Proxy.clientId=$kc_browser_client_id" \
    --set "gateway.oauth2Proxy.cookieDomain=$cookie_domain" \
    --set "gateway.oauth2Proxy.redis.serviceName=$redis_endpoint" \
    --set "gateway.oauth2Proxy.redis.port=$redis_port" \
    --wait \
    --timeout "$HELM_TIMEOUT"

  wait_for_deployment osmo-service "$OSMO_NAMESPACE"
  # Router + UI are deployments within the same release in 6.3.
  wait_for_deployment osmo-router "$OSMO_NAMESPACE" || warn "osmo-router not ready yet"
  wait_for_deployment osmo-ui "$OSMO_NAMESPACE" || warn "osmo-ui not ready yet"

  #----------------------------------------------------------------------------
  # Network Policies
  #----------------------------------------------------------------------------
  section "Deploying Network Policies"

  local network_policy_manifest="$SCRIPT_DIR/manifests/network-policies.yaml"
  if [[ -f "$network_policy_manifest" ]]; then
    ensure_namespace "${OSMO_WORKFLOWS_NAMESPACE:-osmo-workflows}"
    info "Applying Network Policies"
    sed "s|\${OSMO_NAMESPACE}|${OSMO_NAMESPACE}|g; s|\${OSMO_BACKEND_NAMESPACE}|${OSMO_WORKFLOWS_NAMESPACE:-osmo-workflows}|g" \
        "$network_policy_manifest" | kubectl apply -f -
    pass "Network Policies deployed"
  else
    warn "Network Policies manifest not found - consider adding for defense-in-depth"
  fi

  #----------------------------------------------------------------------------
  # Complete
  #----------------------------------------------------------------------------
  section "OSMO Control Plane Deployment Complete"

  info "Service Status:"
  kubectl get pods -n "$OSMO_NAMESPACE"
  echo

  info "Ingress Status:"
  kubectl get ingress -n "$OSMO_NAMESPACE"
  echo

  info "URLs:"
  info "  OSMO: https://$osmo_hostname"
  echo
  info "Admin user: $(tf_get "$tf_outputs" "keycloak_admin_username" "admin")"
  echo
  info "Next: Run 05-deploy-osmo-backend.sh"
}

main "$@"
