#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

# Deploy Keycloak as the OSMO identity provider (IdP)
#
# Follows the OSMO Keycloak setup guide:
#   https://nvidia.github.io/OSMO/release/6.3/deployment_guide/appendix/keycloak_setup.html

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"
# shellcheck source=defaults.conf
source "$SCRIPT_DIR/defaults.conf"

KEYCLOAK_NAMESPACE="keycloak"
KEYCLOAK_HELM_CHART="bitnami/keycloak"
KEYCLOAK_HELM_VERSION="24.4.9"
KEYCLOAK_ADMIN_USER="admin"
KEYCLOAK_DB_NAME="keycloak"

show_help() {
  cat << EOF
Usage: $(basename "$0") [OPTIONS]

Deploy Keycloak as the OSMO identity provider.

Creates a Keycloak database in the existing RDS instance, installs Keycloak
via Helm, and configures the OSMO realm with browser-flow and device clients.

OPTIONS:
    -h, --help              Show this help message
    -t, --tf-dir DIR        Terraform directory (default: $DEFAULT_TF_DIR)
    --admin-password PASS   Keycloak admin password (or set KEYCLOAK_ADMIN_PASSWORD env var)
    --skip-db               Skip database creation (already exists)
    --skip-install          Skip Helm install (already running)

EXAMPLES:
    $(basename "$0") --admin-password 'MySecurePass123!'
    $(basename "$0") --skip-db --skip-install   # only configure realm/clients
EOF
}

tf_dir="$SCRIPT_DIR/$DEFAULT_TF_DIR"
admin_password="${KEYCLOAK_ADMIN_PASSWORD:-}"
skip_db=false
skip_install=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)           show_help; exit 0 ;;
    -t|--tf-dir)         tf_dir="$2"; shift 2 ;;
    --admin-password)    admin_password="$2"; shift 2 ;;
    --skip-db)           skip_db=true; shift ;;
    --skip-install)      skip_install=true; shift ;;
    *)                   fatal "Unknown option: $1" ;;
  esac
done

#------------------------------------------------------------------------------
# Keycloak Admin REST API helpers
#------------------------------------------------------------------------------

kc_get_admin_token() {
  local base_url="$1"
  curl -sS -X POST "${base_url}/realms/master/protocol/openid-connect/token" \
    -d "client_id=admin-cli" \
    -d "username=${KEYCLOAK_ADMIN_USER}" \
    -d "password=${admin_password}" \
    -d "grant_type=password" | jq -r '.access_token'
}

kc_api() {
  local method="$1" url="$2" token="$3"
  shift 3
  curl -sS -X "$method" "$url" \
    -H "Authorization: Bearer $token" \
    -H "Content-Type: application/json" \
    "$@"
}

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------

main() {
  section "Deploying Keycloak (OSMO IdP)"

  require_tools aws kubectl helm jq curl openssl yq

  if [[ -z "$admin_password" ]]; then
    fatal "Keycloak admin password required. Use --admin-password or set KEYCLOAK_ADMIN_PASSWORD."
  fi

  # Read Terraform outputs
  info "Reading Terraform outputs from: $tf_dir"
  local tf_outputs
  tf_outputs=$(read_terraform_outputs "$tf_dir")

  local cluster_name region osmo_hostname osmo_auth_hostname
  local rds_endpoint rds_port rds_username rds_password_secret_arn
  local acm_auth_certificate_arn alb_security_group_id

  cluster_name=$(tf_require "$tf_outputs" "cluster_name" "EKS cluster name")
  region=$(tf_require "$tf_outputs" "aws_region" "AWS region")
  osmo_hostname=$(tf_require "$tf_outputs" "osmo_hostname" "OSMO hostname")
  osmo_auth_hostname=$(tf_require "$tf_outputs" "osmo_auth_hostname" "OSMO auth hostname")
  rds_endpoint=$(tf_require "$tf_outputs" "rds_endpoint" "RDS endpoint")
  rds_port=$(tf_get "$tf_outputs" "rds_port" "5432")
  rds_username=$(tf_get "$tf_outputs" "rds_username" "osmo_admin")
  rds_password_secret_arn=$(tf_require "$tf_outputs" "rds_password_secret_arn" "RDS password secret ARN")
  acm_auth_certificate_arn=$(tf_require "$tf_outputs" "acm_auth_certificate_arn" "ACM auth certificate ARN")
  alb_security_group_id=$(tf_get "$tf_outputs" "alb_security_group_id" "")

  local rds_host="${rds_endpoint%:*}"

  section "Configuration"
  print_kv "Cluster" "$cluster_name"
  print_kv "Region" "$region"
  print_kv "Auth Hostname" "$osmo_auth_hostname"
  print_kv "RDS Host" "$rds_host"
  print_kv "RDS Port" "$rds_port"

  connect_eks "$region" "$cluster_name"
  ensure_namespace "$KEYCLOAK_NAMESPACE"

  #----------------------------------------------------------------------------
  # Step 1: Create Keycloak database in RDS
  #----------------------------------------------------------------------------
  if [[ "$skip_db" != "true" ]]; then
    section "Creating Keycloak Database"

    local db_password
    db_password=$(get_secret_value "$rds_password_secret_arn" "password" "$region")

    info "Creating database '$KEYCLOAK_DB_NAME' in RDS..."
    kubectl delete pod osmo-db-ops -n "$KEYCLOAK_NAMESPACE" --ignore-not-found >/dev/null 2>&1 || true

    kubectl apply -n "$KEYCLOAK_NAMESPACE" -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: osmo-db-ops
spec:
  containers:
    - name: psql
      image: alpine/psql:17.5
      command: ["/bin/sh", "-c"]
      args:
        - |
          PGPASSWORD='${db_password}' psql -U ${rds_username} -h ${rds_host} -p ${rds_port} -d postgres \
            -tc "SELECT 1 FROM pg_database WHERE datname='${KEYCLOAK_DB_NAME}'" | grep -q 1 \
            && echo "Database already exists" \
            || PGPASSWORD='${db_password}' psql -U ${rds_username} -h ${rds_host} -p ${rds_port} -d postgres \
              -c "CREATE DATABASE ${KEYCLOAK_DB_NAME};"
  restartPolicy: Never
EOF

    info "Waiting for DB creation pod to complete..."
    kubectl wait pod/osmo-db-ops -n "$KEYCLOAK_NAMESPACE" --for=condition=Ready --timeout=60s 2>/dev/null || true
    kubectl wait pod/osmo-db-ops -n "$KEYCLOAK_NAMESPACE" --for=jsonpath='{.status.phase}'=Succeeded --timeout=120s || \
      fatal "DB creation pod did not succeed"

    kubectl logs osmo-db-ops -n "$KEYCLOAK_NAMESPACE"
    kubectl delete pod osmo-db-ops -n "$KEYCLOAK_NAMESPACE" --ignore-not-found >/dev/null 2>&1

    info "Creating keycloak-db-secret"
    kubectl create secret generic keycloak-db-secret \
      --namespace "$KEYCLOAK_NAMESPACE" \
      --from-literal=postgres-password="$db_password" \
      --dry-run=client -o yaml | kubectl apply -f -

    pass "Database ready"
  fi

  #----------------------------------------------------------------------------
  # Step 2: Install Keycloak via Helm
  #----------------------------------------------------------------------------
  if [[ "$skip_install" != "true" ]]; then
    section "Installing Keycloak"

    helm_repo_add bitnami https://charts.bitnami.com/bitnami

    local sg_annotation=""
    if [[ -n "$alb_security_group_id" ]]; then
      sg_annotation="alb.ingress.kubernetes.io/security-groups: ${alb_security_group_id}"
    fi

    info "Generating Keycloak Helm values"
    local keycloak_values
    keycloak_values=$(mktemp)

    cat > "$keycloak_values" <<YAML
global:
  security:
    allowInsecureImages: true

image:
  registry: docker.io
  repository: bitnamilegacy/keycloak
  tag: 26.1.1-debian-12-r0

hostname: ${osmo_auth_hostname}
proxy: edge
production: true

tls:
  enabled: true
  autoGenerated: true

auth:
  adminUser: ${KEYCLOAK_ADMIN_USER}
  adminPassword: "${admin_password}"

ingress:
  enabled: true
  tls: true
  ingressClassName: alb
  hostname: ${osmo_auth_hostname}
  annotations:
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/certificate-arn: "${acm_auth_certificate_arn}"
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTPS":443}]'
    alb.ingress.kubernetes.io/ssl-redirect: "443"
    alb.ingress.kubernetes.io/success-codes: "200,302,303"
    # same path the k8s readinessProbe uses — returns 200 on the traffic port
    alb.ingress.kubernetes.io/healthcheck-path: /realms/master
    ${sg_annotation}
  path: /
  pathType: Prefix
  servicePort: 80

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 3
  targetCPU: 80
  targetMemory: 80

resources:
  requests:
    cpu: "500m"
    memory: "512Mi"
  limits:
    cpu: "2"
    memory: "1Gi"

postgresql:
  enabled: false
externalDatabase:
  host: "${rds_host}"
  port: ${rds_port}
  user: "${rds_username}"
  database: "${KEYCLOAK_DB_NAME}"
  existingSecret: "keycloak-db-secret"
  existingSecretPasswordKey: "postgres-password"

extraEnvVars:
  - name: KC_HOSTNAME_STRICT_HTTPS
    value: "true"
  - name: KC_PROXY
    value: "edge"
YAML

    info "Installing Keycloak ${KEYCLOAK_HELM_VERSION}"
    helm upgrade --install keycloak "$KEYCLOAK_HELM_CHART" \
      --version "$KEYCLOAK_HELM_VERSION" \
      --namespace "$KEYCLOAK_NAMESPACE" \
      -f "$keycloak_values" \
      --wait \
      --timeout 600s

    rm -f "$keycloak_values"

    pass "Keycloak installed"
  fi

  #----------------------------------------------------------------------------
  # Step 3: Wait for Keycloak and configure realm/clients via Admin REST API
  #----------------------------------------------------------------------------
  section "Configuring Keycloak for OSMO"

  info "Waiting for Keycloak to become ready..."

  local kc_url=""
  local retries=0
  local max_retries=30

  # Use port-forward to reach Keycloak Admin API
  local pf_pid=""
  local local_port=32080
  info "Starting port-forward to Keycloak..."
  kubectl port-forward svc/keycloak ${local_port}:80 -n "$KEYCLOAK_NAMESPACE" &
  pf_pid=$!
  sleep 5

  kc_url="http://localhost:${local_port}"

  # Wait until Keycloak actually serves (-f so a non-2xx no longer counts as ready).
  while ! curl -fsS -o /dev/null "${kc_url}/realms/master" 2>/dev/null; do
    retries=$((retries + 1))
    if [[ $retries -ge $max_retries ]]; then
      kill "$pf_pid" 2>/dev/null || true
      fatal "Keycloak not reachable after ${max_retries} attempts"
    fi
    sleep 5
  done
  pass "Keycloak is reachable"

  # Obtain admin token
  local token
  token=$(kc_get_admin_token "$kc_url")
  [[ -n "$token" && "$token" != "null" ]] || { kill "$pf_pid" 2>/dev/null || true; fatal "Failed to get admin token"; }

  # Create osmo realm (idempotent: skip if it exists)
  local realm_exists
  realm_exists=$(kc_api GET "${kc_url}/admin/realms/osmo" "$token" -o /dev/null -w "%{http_code}" 2>/dev/null || echo "000")

  if [[ "$realm_exists" != "200" ]]; then
    info "Creating 'osmo' realm..."
    kc_api POST "${kc_url}/admin/realms" "$token" \
      -d '{
        "realm": "osmo",
        "enabled": true,
        "registrationAllowed": false,
        "loginWithEmailAllowed": true,
        "duplicateEmailsAllowed": false,
        "resetPasswordAllowed": true,
        "editUsernameAllowed": false,
        "bruteForceProtected": true,
        "oauth2DeviceCodeLifespan": 600,
        "oauth2DevicePollingInterval": 5,
        "accessTokenLifespan": 3600,
        "ssoSessionIdleTimeout": 1800
      }' >/dev/null
    pass "Realm 'osmo' created"
  else
    info "Realm 'osmo' already exists"
  fi

  # Re-fetch token (may have expired during realm creation)
  token=$(kc_get_admin_token "$kc_url")

  # Create osmo-browser-flow client
  local browser_client_exists
  browser_client_exists=$(kc_api GET "${kc_url}/admin/realms/osmo/clients?clientId=osmo-browser-flow" "$token" | jq 'length')

  local browser_client_secret=""

  if [[ "$browser_client_exists" == "0" ]]; then
    info "Creating 'osmo-browser-flow' client..."
    kc_api POST "${kc_url}/admin/realms/osmo/clients" "$token" \
      -d "{
        \"clientId\": \"osmo-browser-flow\",
        \"name\": \"OSMO Browser Flow\",
        \"enabled\": true,
        \"protocol\": \"openid-connect\",
        \"publicClient\": false,
        \"standardFlowEnabled\": true,
        \"directAccessGrantsEnabled\": false,
        \"serviceAccountsEnabled\": false,
        \"rootUrl\": \"https://${osmo_hostname}\",
        \"baseUrl\": \"https://${osmo_hostname}\",
        \"adminUrl\": \"https://${osmo_hostname}\",
        \"redirectUris\": [\"https://${osmo_hostname}/*\"],
        \"webOrigins\": [\"https://${osmo_hostname}\"],
        \"defaultClientScopes\": [\"openid\", \"email\", \"profile\", \"roles\"]
      }" >/dev/null
    pass "Client 'osmo-browser-flow' created"
  else
    info "Client 'osmo-browser-flow' already exists"
  fi

  # Retrieve browser client secret
  local browser_client_id
  browser_client_id=$(kc_api GET "${kc_url}/admin/realms/osmo/clients?clientId=osmo-browser-flow" "$token" | jq -r '.[0].id')
  browser_client_secret=$(kc_api GET "${kc_url}/admin/realms/osmo/clients/${browser_client_id}/client-secret" "$token" | jq -r '.value')

  # Create osmo-device client (public client for CLI device flow)
  local device_client_exists
  device_client_exists=$(kc_api GET "${kc_url}/admin/realms/osmo/clients?clientId=osmo-device" "$token" | jq 'length')

  if [[ "$device_client_exists" == "0" ]]; then
    info "Creating 'osmo-device' client..."
    kc_api POST "${kc_url}/admin/realms/osmo/clients" "$token" \
      -d "{
        \"clientId\": \"osmo-device\",
        \"name\": \"OSMO Device (CLI)\",
        \"enabled\": true,
        \"protocol\": \"openid-connect\",
        \"publicClient\": true,
        \"standardFlowEnabled\": false,
        \"directAccessGrantsEnabled\": true,
        \"serviceAccountsEnabled\": false,
        \"attributes\": {
          \"oauth2.device.authorization.grant.enabled\": \"true\",
          \"oauth2.device.polling.interval\": \"5\"
        },
        \"rootUrl\": \"https://${osmo_hostname}\",
        \"redirectUris\": [\"https://${osmo_hostname}/*\"],
        \"webOrigins\": [\"https://${osmo_hostname}\"],
        \"defaultClientScopes\": [\"openid\", \"email\", \"profile\", \"roles\"]
      }" >/dev/null
    pass "Client 'osmo-device' created"
  else
    info "Client 'osmo-device' already exists"
  fi

  # Retrieve device client internal ID
  local device_client_id
  device_client_id=$(kc_api GET "${kc_url}/admin/realms/osmo/clients?clientId=osmo-device" "$token" | jq -r '.[0].id')

  #----------------------------------------------------------------------------
  # Step 3b: Create client roles (idempotent — Keycloak returns 409 if exists)
  #----------------------------------------------------------------------------
  info "Ensuring client roles on osmo-browser-flow..."
  for role_spec in \
    "osmo-admin|Admin access to the osmo service" \
    "osmo-user|A regular user of osmo who can submit and query workflows and datasets" \
    "dashboard-admin|Able to make change to the kubernetes dashboard" \
    "dashboard-user|Able to view the kubernetes dashboard" \
    "grafana-admin|" \
    "grafana-user|Able to view dashboards in grafana"; do
    local rname="${role_spec%%|*}" rdesc="${role_spec#*|}"
    kc_api POST "${kc_url}/admin/realms/osmo/clients/${browser_client_id}/roles" "$token" \
      -d "{\"name\":\"${rname}\",\"description\":\"${rdesc}\"}" >/dev/null 2>&1 || true
  done

  info "Ensuring client roles on osmo-device..."
  for role_spec in \
    "osmo-admin|Admin access to the osmo service" \
    "osmo-user|" \
    "osmo-backend|"; do
    local rname="${role_spec%%|*}" rdesc="${role_spec#*|}"
    kc_api POST "${kc_url}/admin/realms/osmo/clients/${device_client_id}/roles" "$token" \
      -d "{\"name\":\"${rname}\",\"description\":\"${rdesc}\"}" >/dev/null 2>&1 || true
  done
  pass "Client roles configured"

  #----------------------------------------------------------------------------
  # Step 3c: Protocol mapper — project client roles into "roles" JWT claim
  #----------------------------------------------------------------------------
  for mapper_pair in "${browser_client_id}|osmo-browser-flow" "${device_client_id}|osmo-device"; do
    local cid="${mapper_pair%%|*}" cname="${mapper_pair#*|}"
    local mapper_exists
    mapper_exists=$(kc_api GET "${kc_url}/admin/realms/osmo/clients/${cid}/protocol-mappers/models" "$token" | \
      jq '[.[] | select(.protocolMapper == "oidc-usermodel-client-role-mapper" and .config["claim.name"] == "roles")] | length')
    if [[ "${mapper_exists:-0}" == "0" ]]; then
      info "Creating 'roles' protocol mapper on ${cname}..."
      kc_api POST "${kc_url}/admin/realms/osmo/clients/${cid}/protocol-mappers/models" "$token" \
        -d "{
          \"name\": \"Create \\\"roles\\\" claim\",
          \"protocol\": \"openid-connect\",
          \"protocolMapper\": \"oidc-usermodel-client-role-mapper\",
          \"consentRequired\": false,
          \"config\": {
            \"multivalued\": \"true\",
            \"userinfo.token.claim\": \"true\",
            \"id.token.claim\": \"true\",
            \"access.token.claim\": \"true\",
            \"claim.name\": \"roles\",
            \"jsonType.label\": \"String\",
            \"usermodel.clientRoleMapping.clientId\": \"${cname}\"
          }
        }" >/dev/null
    else
      info "Protocol mapper 'roles' already exists on ${cname}"
    fi
  done
  pass "Protocol mappers configured"

  token=$(kc_get_admin_token "$kc_url")

  # Create groups: Admin, User, Backend Operator
  for group_name in "Admin" "User" "Backend Operator"; do
    local group_exists
    group_exists=$(kc_api GET "${kc_url}/admin/realms/osmo/groups?search=${group_name// /%20}&exact=true" "$token" | jq 'length')
    if [[ "$group_exists" == "0" ]]; then
      info "Creating group '${group_name}'..."
      kc_api POST "${kc_url}/admin/realms/osmo/groups" "$token" \
        -d "{\"name\": \"${group_name}\"}" >/dev/null
    fi
  done
  pass "Groups configured"

  #----------------------------------------------------------------------------
  # Step 3d: Assign client roles to groups (additive; duplicates are ignored)
  #----------------------------------------------------------------------------
  info "Assigning client roles to groups..."
  token=$(kc_get_admin_token "$kc_url")

  local admin_grp_id user_grp_id backend_grp_id
  admin_grp_id=$(kc_api GET "${kc_url}/admin/realms/osmo/groups?search=Admin&exact=true" "$token" | jq -r '.[0].id')
  user_grp_id=$(kc_api GET "${kc_url}/admin/realms/osmo/groups?search=User&exact=true" "$token" | jq -r '.[0].id')
  backend_grp_id=$(kc_api GET "${kc_url}/admin/realms/osmo/groups?search=Backend%20Operator&exact=true" "$token" | jq -r '.[0].id // empty')

  kc_role_json() {
    kc_api GET "${kc_url}/admin/realms/osmo/clients/$1/roles/$2" "$token"
  }

  local br_admin br_user br_grafana_user br_dashboard_user
  br_admin=$(kc_role_json "$browser_client_id" "osmo-admin")
  br_user=$(kc_role_json "$browser_client_id" "osmo-user")
  br_grafana_user=$(kc_role_json "$browser_client_id" "grafana-user")
  br_dashboard_user=$(kc_role_json "$browser_client_id" "dashboard-user")

  local dv_admin dv_user dv_backend
  dv_admin=$(kc_role_json "$device_client_id" "osmo-admin")
  dv_user=$(kc_role_json "$device_client_id" "osmo-user")
  dv_backend=$(kc_role_json "$device_client_id" "osmo-backend")

  # Admin -> osmo-browser-flow: osmo-admin, osmo-user
  kc_api POST "${kc_url}/admin/realms/osmo/groups/${admin_grp_id}/role-mappings/clients/${browser_client_id}" "$token" \
    -d "[${br_admin},${br_user}]" >/dev/null 2>&1 || true
  # Admin -> osmo-device: osmo-admin, osmo-user
  kc_api POST "${kc_url}/admin/realms/osmo/groups/${admin_grp_id}/role-mappings/clients/${device_client_id}" "$token" \
    -d "[${dv_admin},${dv_user}]" >/dev/null 2>&1 || true
  # User -> osmo-browser-flow: osmo-user, grafana-user, dashboard-user
  kc_api POST "${kc_url}/admin/realms/osmo/groups/${user_grp_id}/role-mappings/clients/${browser_client_id}" "$token" \
    -d "[${br_user},${br_grafana_user},${br_dashboard_user}]" >/dev/null 2>&1 || true
  # User -> osmo-device: osmo-user
  kc_api POST "${kc_url}/admin/realms/osmo/groups/${user_grp_id}/role-mappings/clients/${device_client_id}" "$token" \
    -d "[${dv_user}]" >/dev/null 2>&1 || true
  # Backend Operator -> osmo-device: osmo-backend
  if [[ -n "$backend_grp_id" && "$backend_grp_id" != "null" ]]; then
    kc_api POST "${kc_url}/admin/realms/osmo/groups/${backend_grp_id}/role-mappings/clients/${device_client_id}" "$token" \
      -d "[${dv_backend}]" >/dev/null 2>&1 || true
  fi

  pass "Group-to-role bindings configured"

  # Create admin user in the osmo realm
  local keycloak_admin_username
  keycloak_admin_username=$(tf_get "$tf_outputs" "keycloak_admin_username" "admin")

  local user_exists
  user_exists=$(kc_api GET "${kc_url}/admin/realms/osmo/users?username=${keycloak_admin_username}&exact=true" "$token" | jq 'length')

  if [[ "$user_exists" == "0" ]]; then
    info "Creating OSMO admin user '${keycloak_admin_username}' in osmo realm..."
    kc_api POST "${kc_url}/admin/realms/osmo/users" "$token" \
      -d "{
        \"username\": \"${keycloak_admin_username}\",
        \"enabled\": true,
        \"emailVerified\": true,
        \"credentials\": [{
          \"type\": \"password\",
          \"value\": \"${admin_password}\",
          \"temporary\": true
        }]
      }" >/dev/null

    # Add user to Admin group
    local user_id admin_group_id
    user_id=$(kc_api GET "${kc_url}/admin/realms/osmo/users?username=${keycloak_admin_username}&exact=true" "$token" | jq -r '.[0].id')
    admin_group_id=$(kc_api GET "${kc_url}/admin/realms/osmo/groups?search=Admin&exact=true" "$token" | jq -r '.[0].id')
    kc_api PUT "${kc_url}/admin/realms/osmo/users/${user_id}/groups/${admin_group_id}" "$token" >/dev/null
    pass "Admin user created and added to Admin group"
  else
    info "Admin user '${keycloak_admin_username}' already exists"
  fi

  # Stop port-forward
  kill "$pf_pid" 2>/dev/null || true
  wait "$pf_pid" 2>/dev/null || true

  #----------------------------------------------------------------------------
  # Step 4: Store browser client secret for OSMO's OAuth2 Proxy
  #----------------------------------------------------------------------------
  section "Storing OAuth2 Proxy Secrets"

  local osmo_namespace="${OSMO_NAMESPACE:-osmo}"
  ensure_namespace "$osmo_namespace"

  if [[ -n "$browser_client_secret" && "$browser_client_secret" != "null" ]]; then
    # cookie_secret must survive re-runs: it keys the oauth2-proxy session cookies
    # already in users' browsers, so a new value logs everyone out. Reuse the stored
    # one and generate only on first install. (client_secret above is re-read from
    # Keycloak every run on purpose — Keycloak is authoritative for it.)
    # Kept base64 in the variable and decoded to a file: the value is raw bytes and
    # command substitution cannot carry NUL. openssl base64 for BSD/GNU portability.
    local cookie_b64 cookie_file
    cookie_b64=$(kubectl get secret oauth2-proxy-secrets \
      --namespace "$osmo_namespace" \
      -o jsonpath='{.data.cookie_secret}' 2>/dev/null || true)
    cookie_file=$(mktemp)

    if [[ -n "$cookie_b64" ]]; then
      printf '%s' "$cookie_b64" | openssl base64 -d -A > "$cookie_file"
      info "Reusing existing oauth2-proxy cookie_secret (rotating it would sign out all users)"
    else
      openssl rand 32 > "$cookie_file"
      info "Generating oauth2-proxy cookie_secret"
    fi

    # Guard against writing an empty/short key — oauth2-proxy needs 16, 24 or 32 bytes.
    if [[ ! -s "$cookie_file" ]]; then
      rm -f "$cookie_file"
      fatal "Failed to obtain a cookie_secret for oauth2-proxy"
    fi

    info "Creating oauth2-proxy-secrets in namespace $osmo_namespace"
    kubectl create secret generic oauth2-proxy-secrets \
      --namespace "$osmo_namespace" \
      --from-literal=client_secret="$browser_client_secret" \
      --from-file=cookie_secret="$cookie_file" \
      --dry-run=client -o yaml | kubectl apply -f -
    rm -f "$cookie_file"
    pass "OAuth2 Proxy secrets created"
  else
    warn "Could not retrieve browser client secret. Create oauth2-proxy-secrets manually."
  fi

  #----------------------------------------------------------------------------
  # Complete
  #----------------------------------------------------------------------------
  section "Keycloak Deployment Complete"

  info "Keycloak:"
  print_kv "URL" "https://${osmo_auth_hostname}"
  print_kv "Admin Console" "https://${osmo_auth_hostname}/admin/master/console/"
  print_kv "Realm" "osmo"
  print_kv "Browser Client" "osmo-browser-flow"
  print_kv "Device Client" "osmo-device"
  echo
  info "OIDC Endpoints:"
  print_kv "Issuer" "https://${osmo_auth_hostname}/realms/osmo"
  print_kv "JWKS" "https://${osmo_auth_hostname}/realms/osmo/protocol/openid-connect/certs"
  print_kv "Device Auth" "https://${osmo_auth_hostname}/realms/osmo/protocol/openid-connect/auth/device"
  echo
  info "Next: Run 04-deploy-osmo-control-plane.sh"
}

main "$@"
