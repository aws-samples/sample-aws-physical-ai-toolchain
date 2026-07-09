#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

# Deploy OSMO Backend Operator

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

Deploy OSMO Backend Operator for workload execution.

OPTIONS:
    -h, --help              Show this help message
    -t, --tf-dir DIR        Terraform directory (default: $DEFAULT_TF_DIR)
    --service-url URL       OSMO service URL (auto-detected if not provided)
    --backend-name NAME     Backend name (default: default; must exist in the ConfigMap backends)
    --config-preview        Print configuration and exit

EXAMPLES:
    $(basename "$0")
    $(basename "$0") --service-url https://osmo.example.com
    $(basename "$0") --backend-name my-backend
EOF
}

# Default values
tf_dir="$SCRIPT_DIR/$DEFAULT_TF_DIR"
service_url=""
# Must match a backend defined in the ConfigMap (services.configs.backends.*)
# and referenced by the pool (services.configs.pools.default.backend). In
# ConfigMap mode the agent cannot auto-create unknown backends, so the operator
# must register as an existing one ("default").
backend_name="default"
config_preview=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)          show_help; exit 0 ;;
    -t|--tf-dir)        tf_dir="$2"; shift 2 ;;
    --service-url)      service_url="$2"; shift 2 ;;
    --backend-name)     backend_name="$2"; shift 2 ;;
    --config-preview)   config_preview=true; shift ;;
    *)                  fatal "Unknown option: $1" ;;
  esac
done

detect_service_url() {
  local tf="$1"
  local hostname
  hostname=$(tf_require "$tf" "osmo_hostname" "OSMO hostname")
  echo "https://${hostname}"
}

create_backend_token() {
  local namespace="$1"
  local admin_user="$2"
  local secret_name="osmo-operator-token"

  # Check if OSMO service is ready
  if ! kubectl get deployment osmo-service -n "$OSMO_NAMESPACE" &>/dev/null; then
    fatal "OSMO service not deployed. Run 04-deploy-osmo-control-plane.sh first."
  fi

  # Check if a valid token already exists
  local existing_token
  existing_token=$(kubectl get secret "$secret_name" -n "$namespace" \
    -o jsonpath='{.data.token}' 2>/dev/null | base64 -d 2>/dev/null || echo "")
  if [[ -n "$existing_token" && "$existing_token" != "placeholder-update-via-osmo-cli" ]]; then
    info "Using existing token from secret $secret_name"
    return 0
  fi

  info "Creating backend operator service token via OSMO API..."

  local token_name
  token_name="backend-token-$(date -u +%Y%m%d%H%M%S)"
  local expiry_date
  expiry_date=$(date -u -d "+1 year" +%F 2>/dev/null || date -u -v+1y +%F 2>/dev/null || echo "2027-01-01")

  # Clean up any stale port-forward this script may have left on a prior failed
  # run (fuser is often not installed, so match the process directly).
  pkill -f "kubectl port-forward.*deploy/osmo-service" 2>/dev/null || true

  # Pick a free local port to avoid colliding with an orphaned forward.
  local pf_port
  pf_port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()' 2>/dev/null || echo 18000)

  # Port-forward directly to the OSMO service container (port 8000, HTTPS),
  # bypassing the gateway. During bootstrap the service trusts the
  # x-osmo-user / x-osmo-roles headers we set below.
  info "Starting port-forward to osmo-service (direct, https://localhost:${pf_port} -> 8000)..."
  kubectl port-forward -n "$OSMO_NAMESPACE" deploy/osmo-service "${pf_port}:8000" &>/dev/null &
  local pf_pid=$!
  # Clean up the port-forward whether the function returns normally or the
  # script exits early via fatal() (RETURN alone does not fire on exit).
  # shellcheck disable=SC2064
  trap "kill $pf_pid 2>/dev/null || true; wait $pf_pid 2>/dev/null || true" RETURN EXIT

  local osmo_api="https://localhost:${pf_port}"

  local max_wait=30 elapsed=0
  while true; do
    local status_code
    status_code=$(curl -sk -o /dev/null -w "%{http_code}" "${osmo_api}/api/version" 2>/dev/null || echo "000")
    if [[ "$status_code" =~ ^(200|401|403)$ ]]; then
      break
    fi
    sleep 1
    elapsed=$((elapsed + 1))
    if [[ $elapsed -ge $max_wait ]]; then
      fatal "Port-forward to osmo-service failed to start within ${max_wait}s"
    fi
  done
  info "Port-forward ready"

  local auth_headers=(-H "x-osmo-user: ${admin_user}" -H "x-osmo-roles: osmo-admin")

  # Ensure admin user exists
  info "Bootstrapping admin user '${admin_user}' with osmo-admin and osmo-backend roles..."
  local create_response create_code
  create_response=$(curl -sk -w "\n%{http_code}" -X POST \
    "${osmo_api}/api/auth/user" \
    "${auth_headers[@]}" \
    -H "Content-Type: application/json" \
    -d "{\"id\": \"${admin_user}\"}")
  create_code=$(echo "$create_response" | tail -1)
  if [[ "$create_code" == "200" || "$create_code" == "201" ]]; then
    info "Admin user created"
  elif echo "$create_response" | grep -q "already exists"; then
    info "Admin user already exists"
  else
    fatal "Failed to create admin user (HTTP $create_code): $(echo "$create_response" | sed '$d')"
  fi

  # Always assign roles (the create endpoint ignores the roles array,
  # and the roles POST is idempotent so this is safe to run every time)
  for role in osmo-admin osmo-backend; do
    curl -sk -X POST "${osmo_api}/api/auth/user/${admin_user}/roles" \
      "${auth_headers[@]}" \
      -H "Content-Type: application/json" \
      -d "{\"role_name\": \"${role}\"}" >/dev/null
  done
  pass "Admin roles assigned (osmo-admin, osmo-backend)"

  # Create access token for the backend operator
  info "Creating service token: $token_name (expires: $expiry_date)..."
  local token_response http_code token_body
  token_response=$(curl -sk -w "\n%{http_code}" -X POST \
    "${osmo_api}/api/auth/user/${admin_user}/access_token/${token_name}?expires_at=${expiry_date}&roles=osmo-backend" \
    "${auth_headers[@]}" \
    -H "Content-Type: application/json")

  http_code=$(echo "$token_response" | tail -1)
  token_body=$(echo "$token_response" | sed '$d')

  if [[ "$http_code" != "200" && "$http_code" != "201" ]]; then
    fatal "Token creation API returned HTTP $http_code: $token_body"
  fi

  local service_token
  service_token=$(echo "$token_body" | jq -r '. // empty' 2>/dev/null || echo "")
  if [[ -z "$service_token" ]]; then
    service_token=$(echo "$token_body" | tr -d '"' | tr -d '\r' | xargs)
  fi

  if [[ -z "$service_token" ]]; then
    fatal "Failed to extract service token from API response"
  fi
  pass "Service token created: $token_name (expires: $expiry_date)"

  # Store token in Kubernetes secret
  kubectl create secret generic "$secret_name" \
    --namespace "$namespace" \
    --from-literal=token="$service_token" \
    --dry-run=client -o yaml | kubectl apply -f -
  pass "Token stored in secret $secret_name"
}

main() {
  section "Deploying OSMO Backend Operator"

  require_tools kubectl helm jq curl

  # Read Terraform outputs
  info "Reading Terraform outputs from: $tf_dir"
  local tf_outputs
  tf_outputs=$(read_terraform_outputs "$tf_dir")

  local cluster_name region deployment_mode
  cluster_name=$(tf_require "$tf_outputs" "cluster_name" "EKS cluster name")
  region=$(tf_require "$tf_outputs" "aws_region" "AWS region")
  deployment_mode=$(tf_get "$tf_outputs" "deployment_mode" "full")

  # Admin username comes from Terraform (keycloak_admin_username) — the same
  # source 03-deploy-keycloak.sh uses to create the realm user — so the OSMO
  # DB bootstrap and the Keycloak user always match.
  local admin_user
  admin_user=$(tf_get "$tf_outputs" "keycloak_admin_username" "admin")

  # Check deployment mode
  if ! should_deploy_backend "$deployment_mode"; then
    info "Deployment mode is '$deployment_mode' - skipping Backend Operator"
    exit 0
  fi

  # Get external service URL for backend-only mode
  local external_service_url
  external_service_url=$(tf_get "$tf_outputs" "external_service_url" "")

  # S3 configuration
  local s3_workflows_bucket
  s3_workflows_bucket=$(tf_require "$tf_outputs" "s3_workflows_bucket_name" "S3 workflows bucket")

  # IRSA role
  local osmo_backend_role_arn
  osmo_backend_role_arn=$(tf_require "$tf_outputs" "osmo_backend_role_arn" "OSMO backend role ARN")

  # Workflow-pod IRSA (task pods authenticate to S3 for dataset I/O via this SA)
  local osmo_workflow_role_arn osmo_workflow_sa
  osmo_workflow_role_arn=$(tf_require "$tf_outputs" "osmo_workflow_role_arn" "OSMO workflow role ARN")
  osmo_workflow_sa=$(tf_get "$tf_outputs" "osmo_workflow_service_account" "osmo-workflow")

  # Connect to cluster
  connect_eks "$region" "$cluster_name"

  # Determine service URL
  if [[ -z "$service_url" ]]; then
    if [[ -n "$external_service_url" ]]; then
      service_url="$external_service_url"
    else
      service_url=$(detect_service_url "$tf_outputs")
    fi
  fi

  # Print configuration
  section "Configuration"
  print_kv "Cluster" "$cluster_name"
  print_kv "Region" "$region"
  print_kv "Deployment Mode" "$deployment_mode"
  print_kv "Service URL" "$service_url"
  print_kv "Backend Name" "$backend_name"
  print_kv "Operator Namespace" "$OSMO_OPERATOR_NAMESPACE"
  print_kv "Workflows Namespace" "$OSMO_WORKFLOWS_NAMESPACE"
  print_kv "S3 Workflows Bucket" "$s3_workflows_bucket"
  print_kv "OSMO Backend Role" "$osmo_backend_role_arn"

  if [[ "$config_preview" == "true" ]]; then
    info "Config preview mode - exiting"
    exit 0
  fi

  # Create namespaces
  ensure_namespace "$OSMO_OPERATOR_NAMESPACE"
  ensure_namespace "$OSMO_WORKFLOWS_NAMESPACE"

  # Create the workflow-pod ServiceAccount with the IRSA annotation. Workflow
  # task pods (set via the pod template's spec.serviceAccountName) run under
  # this SA so osmo-ctrl authenticates to S3 via the IRSA web-identity token
  # instead of falling back to the EKS node instance role.
  info "Creating workflow ServiceAccount '$osmo_workflow_sa' in $OSMO_WORKFLOWS_NAMESPACE (IRSA)..."
  kubectl create serviceaccount "$osmo_workflow_sa" -n "$OSMO_WORKFLOWS_NAMESPACE" \
    --dry-run=client -o yaml | kubectl apply -f -
  kubectl annotate serviceaccount "$osmo_workflow_sa" -n "$OSMO_WORKFLOWS_NAMESPACE" \
    "eks.amazonaws.com/role-arn=$osmo_workflow_role_arn" --overwrite
  pass "Workflow ServiceAccount ready: $osmo_workflow_sa -> $osmo_workflow_role_arn"

  # Bootstrap admin user and create backend operator token
  create_backend_token "$OSMO_OPERATOR_NAMESPACE" "$admin_user"

  # Deploy Backend Operator
  section "Deploying OSMO Backend Operator"

  helm_repo_add "$OSMO_HELM_REPO_NAME" "$OSMO_HELM_REPO"

  helm_upgrade_install osmo-operator "$OSMO_HELM_REPO_NAME/backend-operator" \
    "$OSMO_OPERATOR_NAMESPACE" \
    --version "$OSMO_CHART_VERSION" \
    -f "$SCRIPT_DIR/values/osmo-backend-operator.yaml" \
    --set global.osmoImageTag="$OSMO_VERSION" \
    --set global.serviceUrl="$service_url" \
    --set global.agentNamespace="$OSMO_OPERATOR_NAMESPACE" \
    --set global.backendNamespace="$OSMO_WORKFLOWS_NAMESPACE" \
    --set global.backendName="$backend_name" \
    --timeout "$HELM_TIMEOUT"

  # The backend-operator chart creates its listener/worker ServiceAccounts but
  # does not expose SA annotations, so apply the IRSA role-arn annotation here.
  # SA names are "<release>-backend-{listener,worker}" (release: osmo-operator).
  info "Annotating backend operator ServiceAccounts for IRSA..."
  for sa in osmo-operator-backend-listener osmo-operator-backend-worker; do
    kubectl annotate serviceaccount "$sa" -n "$OSMO_OPERATOR_NAMESPACE" \
      "eks.amazonaws.com/role-arn=$osmo_backend_role_arn" --overwrite 2>/dev/null \
      || warn "Could not annotate SA $sa (not created yet?)"
  done
  # Restart so the pods pick up the projected web-identity token
  kubectl rollout restart deploy -n "$OSMO_OPERATOR_NAMESPACE" 2>/dev/null || true

  # Wait for backend operator to be ready (deployment names include Helm release prefix)
  wait_for_deployment osmo-operator-osmo-backend-listener "$OSMO_OPERATOR_NAMESPACE"
  wait_for_deployment osmo-operator-osmo-backend-worker "$OSMO_OPERATOR_NAMESPACE"

  section "OSMO Backend Operator Deployment Complete"

  # Show status
  info "Backend Operator Status:"
  kubectl get pods -n "$OSMO_OPERATOR_NAMESPACE"
  echo

  info "Workflows Namespace ($OSMO_WORKFLOWS_NAMESPACE):"
  kubectl get pods -n "$OSMO_WORKFLOWS_NAMESPACE" 2>/dev/null || info "No workloads running yet"
  echo

  info "Next steps:"
  info "  Deployment is complete. In 6.3 (ConfigMap mode) all configuration is"
  info "  declarative and already applied by 03 — scripts 05-09 are retired no-ops."
  info "  1. Verify the pool is ONLINE:"
  info "       osmo pool list   (or check the UI at https://<osmo-hostname>)"
  info "  2. Register the workflow credential:"
  info "       osmo credential set huggingface_token --type GENERIC --payload token=<hf-token>"
  info "  3. Submit a workflow:"
  info "       osmo workflow submit 003-workflows/cosmos_transfer.yaml"
}

main "$@"
