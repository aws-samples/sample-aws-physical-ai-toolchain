#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

# Common shell functions for OSMO on AWS deployment scripts
_COMMON_SH_LOADED=1

#------------------------------------------------------------------------------
# Logging Functions
#------------------------------------------------------------------------------

if [[ -z "${NO_COLOR+x}" ]]; then
  info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*" >&2; }
  warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*" >&2; }
  error() { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; }
  pass()  { printf '\033[1;32m[PASS]\033[0m  %s\n' "$*" >&2; }
else
  info()  { printf '[INFO]  %s\n' "$*" >&2; }
  warn()  { printf '[WARN]  %s\n' "$*" >&2; }
  error() { printf '[ERROR] %s\n' "$*" >&2; }
  pass()  { printf '[PASS]  %s\n' "$*" >&2; }
fi

fatal() {
  error "$@"
  exit 1
}

section() {
  echo
  echo "============================================"
  echo "$*"
  echo "============================================"
}

print_kv() {
  printf '  %-24s %s\n' "$1:" "$2"
}

#------------------------------------------------------------------------------
# Tool Validation
#------------------------------------------------------------------------------

require_tools() {
  local missing=()
  for tool in "$@"; do
    if ! command -v "$tool" &>/dev/null; then
      missing+=("$tool")
    fi
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    fatal "Missing required tools: ${missing[*]}"
  fi
}

#------------------------------------------------------------------------------
# Terraform Integration
#------------------------------------------------------------------------------

read_terraform_outputs() {
  local tf_dir="${1:?terraform directory required}"

  [[ -d "$tf_dir" ]] || fatal "Terraform directory not found: $tf_dir"

  local output
  if output=$(cd "$tf_dir" && terraform output -json 2>/dev/null) && [[ -n "$output" && "$output" != "{}" ]]; then
    echo "$output"
  else
    fatal "Unable to read terraform outputs from $tf_dir. Ensure 'terraform apply' has been run and state is accessible."
  fi
}

tf_get() {
  local json="${1:?json required}"
  local key="${2:?key required}"
  local default="${3:-}"

  local val
  val=$(echo "$json" | jq -r ".$key.value // empty" 2>/dev/null)

  if [[ -n "$val" && "$val" != "null" ]]; then
    echo "$val"
  elif [[ -n "$default" ]]; then
    echo "$default"
  fi
}

tf_require() {
  local json="${1:?json required}"
  local key="${2:?key required}"
  local description="${3:-$key}"

  local val
  val=$(tf_get "$json" "$key")

  [[ -n "$val" ]] || fatal "$description not found in terraform outputs (key: $key)"
  echo "$val"
}

#------------------------------------------------------------------------------
# Kubernetes Helpers
#------------------------------------------------------------------------------

connect_eks() {
  local region="${1:?region required}"
  local cluster_name="${2:?cluster name required}"

  info "Connecting to EKS cluster: $cluster_name"
  aws eks update-kubeconfig --region "$region" --name "$cluster_name" || \
    fatal "Failed to update kubeconfig"

  verify_cluster_connectivity
}

verify_cluster_connectivity() {
  info "Verifying cluster connectivity..."

  if ! kubectl cluster-info &>/dev/null; then
    error "Cannot connect to Kubernetes cluster"
    error "Possible causes:"
    error "  - EKS cluster not ready"
    error "  - VPN connection required (if private endpoint only)"
    error "  - IAM permissions insufficient"
    fatal "Cluster connectivity check failed"
  fi

  pass "Cluster connectivity verified"
}

ensure_namespace() {
  local ns="${1:?namespace required}"
  kubectl create namespace "$ns" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
}

wait_for_deployment() {
  local name="${1:?deployment name required}"
  local namespace="${2:?namespace required}"
  local timeout="${3:-300s}"

  info "Waiting for deployment $name in $namespace..."
  kubectl rollout status deployment/"$name" -n "$namespace" --timeout="$timeout" || \
    fatal "Deployment $name failed to become ready"
}

wait_for_pods() {
  local selector="${1:?selector required}"
  local namespace="${2:?namespace required}"
  local timeout="${3:-300}"

  info "Waiting for pods with selector: $selector"

  local start_time
  start_time=$(date +%s)

  while true; do
    local ready
    ready=$(kubectl get pods -n "$namespace" -l "$selector" -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "")

    if [[ -n "$ready" ]] && [[ ! "$ready" =~ "False" ]]; then
      pass "Pods ready"
      return 0
    fi

    local elapsed
    elapsed=$(($(date +%s) - start_time))
    if [[ $elapsed -gt $timeout ]]; then
      fatal "Timeout waiting for pods: $selector"
    fi

    sleep 5
  done
}

#------------------------------------------------------------------------------
# Helm Helpers
#------------------------------------------------------------------------------

helm_repo_add() {
  local name="${1:?repo name required}"
  local url="${2:?repo url required}"

  if [[ "$url" == oci://* ]]; then
    info "OCI registry: $url (no repo add needed)"
    return 0
  fi

  # Always (re-)add with --force-update so a pre-existing repo registered under
  # the same name but a different/stale URL (e.g. an old nvstaging endpoint) is
  # corrected to the configured URL instead of being silently kept.
  local existing_url
  existing_url=$(helm repo list -o json 2>/dev/null \
    | jq -r --arg n "$name" '.[] | select(.name==$n) | .url' 2>/dev/null || true)
  if [[ -n "$existing_url" && "$existing_url" != "$url" ]]; then
    warn "Helm repo '$name' has stale URL ($existing_url); updating to $url"
  else
    info "Adding/updating Helm repo: $name -> $url"
  fi
  helm repo add "$name" "$url" --force-update
  helm repo update "$name" 2>/dev/null || true
}

helm_upgrade_install() {
  local release="${1:?release name required}"
  local chart="${2:?chart required}"
  local namespace="${3:?namespace required}"
  shift 3

  info "Installing/upgrading Helm release: $release"

  helm upgrade --install "$release" "$chart" \
    --namespace "$namespace" \
    --create-namespace \
    --wait \
    "$@"
}

#------------------------------------------------------------------------------
# AWS Helpers
#------------------------------------------------------------------------------

get_secret_value() {
  local secret_arn="${1:?secret ARN required}"
  local key="${2:-}"
  local region="${3:-}"

  local region_args=()
  if [[ -n "$region" ]]; then
    region_args=(--region "$region")
  fi

  local secret_json
  secret_json=$(aws secretsmanager get-secret-value --secret-id "$secret_arn" "${region_args[@]}" --query SecretString --output text) || \
    fatal "Failed to retrieve secret: $secret_arn"

  local result
  if [[ -n "$key" ]]; then
    result=$(echo "$secret_json" | jq -r ".$key // empty")
  else
    result="$secret_json"
  fi

  if [[ -z "$result" ]]; then
    fatal "Secret value is empty (secret: $secret_arn, key: ${key:-<whole secret>})"
  fi

  echo "$result"
}

create_k8s_secret_from_aws() {
  local aws_secret_arn="${1:?AWS secret ARN required}"
  local k8s_secret_name="${2:?K8s secret name required}"
  local namespace="${3:?namespace required}"
  local region="${4:-}"

  info "Creating K8s secret from AWS Secrets Manager: $k8s_secret_name"

  local region_args=()
  if [[ -n "$region" ]]; then
    region_args=(--region "$region")
  fi

  local secret_json
  secret_json=$(aws secretsmanager get-secret-value --secret-id "$aws_secret_arn" "${region_args[@]}" --query SecretString --output text) || \
    fatal "Failed to retrieve secret: $aws_secret_arn"

  # Convert JSON to --from-literal args
  local args=()
  while IFS='=' read -r key value; do
    args+=("--from-literal=$key=$value")
  done < <(echo "$secret_json" | jq -r 'to_entries | .[] | "\(.key)=\(.value)"')

  kubectl create secret generic "$k8s_secret_name" \
    --namespace "$namespace" \
    "${args[@]}" \
    --dry-run=client -o yaml | kubectl apply -f -
}

#------------------------------------------------------------------------------
# Configuration Helpers
#------------------------------------------------------------------------------

apply_template() {
  local template="${1:?template file required}"
  local output="${2:-}"

  if [[ ! -f "$template" ]]; then
    fatal "Template not found: $template"
  fi

  if [[ -n "$output" ]]; then
    envsubst < "$template" > "$output"
    info "Generated: $output"
  else
    envsubst < "$template"
  fi
}

generate_mek_config() {
  local secret_name="${1:-osmo-mek}"
  local namespace="${2:-osmo}"

  local encoded
  # Reuse existing MEK if the ConfigMap already exists (regenerating would
  # invalidate all data encrypted with the previous key).
  encoded=$(kubectl get configmap "$secret_name" -n "$namespace" \
    -o jsonpath='{.data.mek\.yaml}' 2>/dev/null \
    | grep 'key1:' | awk '{print $2}' || true)

  if [[ -z "$encoded" ]]; then
    local key jwk
    key=$(openssl rand -base64 32 | tr -d '\n')
    jwk="{\"k\":\"${key}\",\"kid\":\"key1\",\"kty\":\"oct\"}"
    encoded=$(echo -n "$jwk" | base64 | tr -d '\n')
    info "Generated new MEK"
  else
    info "Reusing existing MEK from ConfigMap $secret_name"
  fi

  cat << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: ${secret_name}
  namespace: ${namespace}
data:
  mek.yaml: |
    currentMek: key1
    meks:
      key1: ${encoded}
EOF
}

#------------------------------------------------------------------------------
# Deployment Mode Helpers
#------------------------------------------------------------------------------

get_deployment_mode() {
  local tf_outputs="${1:?terraform outputs required}"
  tf_get "$tf_outputs" "deployment_mode" "full"
}

should_deploy_control_plane() {
  local mode="${1:?deployment mode required}"
  [[ "$mode" == "full" || "$mode" == "control-plane-only" ]]
}

should_deploy_backend() {
  local mode="${1:?deployment mode required}"
  [[ "$mode" == "full" || "$mode" == "backend-only" ]]
}

should_deploy_gpu_operator() {
  local mode="${1:?deployment mode required}"
  [[ "$mode" == "full" || "$mode" == "backend-only" ]]
}

#------------------------------------------------------------------------------
# Central Configuration (deployment-config.yaml)
#------------------------------------------------------------------------------

read_config() {
  local key="${1:?config key required}"
  local config="${DEPLOYMENT_CONFIG:-}"

  if [[ -z "$config" ]]; then
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd)"
    config="$script_dir/config/deployment-config.yaml"
  fi

  if [[ ! -f "$config" ]]; then
    echo ""
    return
  fi

  # mikefarah yq (v4) does not support jq's `// empty`; it returns the literal
  # string "null" for a missing key. Normalize that to an empty string so
  # callers' `${VAR:-fallback}` defaulting works as intended.
  local val
  val="$(yq ".$key" "$config" 2>/dev/null)"
  if [[ "$val" == "null" || -z "$val" ]]; then
    echo ""
  else
    echo "$val"
  fi
}

#------------------------------------------------------------------------------
# Helm Chart Fetch (for pre-release RC charts from NGC)
#------------------------------------------------------------------------------

fetch_osmo_chart() {
  local chart_name="${1:?chart name required}"
  local chart_version="${OSMO_CHART_VERSION:-$(read_config 'osmo.chart_version')}"
  local base_url="${OSMO_CHART_BASE_URL:-$(read_config 'osmo.chart_base_url')}"
  local cache_dir
  cache_dir="$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd)/../.chart-cache"
  local tgz="${cache_dir}/${chart_name}-${chart_version}.tgz"

  mkdir -p "$cache_dir"
  if [[ ! -f "$tgz" ]]; then
    info "Fetching ${chart_name}-${chart_version}.tgz from NGC..."
    helm fetch "${base_url}/${chart_name}-${chart_version}.tgz" -d "$cache_dir" || \
      fatal "Failed to fetch chart: ${chart_name}-${chart_version}"
  else
    info "Using cached chart: ${chart_name}-${chart_version}.tgz"
  fi
  echo "$tgz"
}
