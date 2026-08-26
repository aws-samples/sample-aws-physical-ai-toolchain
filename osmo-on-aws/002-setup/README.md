# OSMO on AWS - Setup Scripts

This directory contains scripts for deploying OSMO components on EKS.

## Prerequisites

Before running these scripts:

1. Complete infrastructure deployment in `../001-iac/`
2. Have `kubectl`, `helm`, and `aws` CLI configured
3. Ensure AWS credentials have appropriate permissions

## Deployment Order

Run scripts in numerical order:

```bash
# 1. Deploy AWS prerequisites (LB Controller, External Secrets, Storage Classes)
./01-deploy-aws-prerequisites.sh

# 2. Deploy GPU Infrastructure: GPU Operator + KAI Scheduler (skip for control-plane-only mode)
./02-deploy-gpu-infrastructure.sh

# 3. Deploy Keycloak (identity provider)
./03-deploy-keycloak.sh --admin-password 'YourSecurePassword!'

# 4. Deploy OSMO Control Plane — single consolidated `service` chart
#    (API + Router + UI + gateway) and apply declarative ConfigMap config.
./04-deploy-osmo-control-plane.sh --ngc-api-key "your-ngc-key"

# 5. Deploy OSMO Backend Operator (also creates the osmo-workflow IRSA SA)
./05-deploy-osmo-backend.sh
```

> **6.3 / ConfigMap mode:** Scripts `05`–`09` are **retired no-op stubs**. All of
> their configuration (pools, `l40s` platform, pod templates, roles, KAI backend
> scheduler, dataset/workflow storage) is now declarative under `services.configs.*`
> in `values/osmo-control-plane.yaml` (+ a generated overlay) and applied by Step 4.
> S3 access uses **IRSA**, not registered credentials. To change config, edit the
> values file and re-run `03`.

## Deployment Modes

| Mode | Scripts to Run |
|------|----------------|
| Full | 01, 02-gpu, 02-keycloak, 03, 04 |
| Control-Plane-Only | 01, 02-keycloak, 03 |
| Backend-Only | 01, 02-gpu, 04 (with --service-url) |

## Email anti-spoofing (DNS)

Email anti-spoofing records are managed by Terraform in `../001-iac`, not by these
setup scripts. For these non-mail-sending domains it publishes DMARC (`p=reject`)
on the zone apex **and** both OSMO hostnames, plus SPF (`v=spf1 -all`) on the apex.
DMARC `p=reject` already blocks spoofing from every one of these names, so the
default configuration is complete out of the box — apply it with
`cd ../001-iac && terraform apply`.

For the two OSMO **hostnames** we deliberately rely on DMARC alone and publish no
per-host SPF: DMARC `p=reject` fully blocks spoofing there, so a per-host SPF
record would add nothing. See `001-iac/variables.tf`
(`enable_email_spoofing_protection`, `dmarc_report_address`).

## Configuration

### defaults.conf

Contains default versions, namespaces, and Helm repository URLs. Override with environment variables:

```bash
export OSMO_VERSION="6.3.1"
export OSMO_NAMESPACE="my-osmo"
./04-deploy-osmo-control-plane.sh
```

### values/

Helm values files for each component. Customize before deployment:

- `osmo-control-plane.yaml` - Consolidated `service` chart (API + Router + UI + gateway), **plus the declarative `services.configs.*` block** (pools, platforms, pod templates, roles, backend scheduler) that is the source of truth in ConfigMap mode
- `osmo-backend-operator.yaml` - Backend Listener and Worker
- `nvidia-gpu-operator.yaml` - GPU Operator settings
- `aws-load-balancer-controller.yaml` - ALB Controller settings

> The standalone `osmo-router.yaml` and `osmo-web-ui.yaml` values were removed in
> 6.3 — router and UI are part of the `service` chart.

### config/

The legacy `*.template.json` files (service/workflow/scheduler/dataset/gpu-platform/
gpu-pod-template) drove the retired `05`–`09` API-config scripts. In 6.3 their
content lives in `values/osmo-control-plane.yaml` under `services.configs.*`. The
dynamic pieces (bucket paths, workflow storage URLs, ECR registry) are rendered by
`04-deploy-osmo-control-plane.sh` into `config/out/configs-overlay.yaml` and layered
via `helm -f`.

### manifests/

Kubernetes manifests:

- `storage-class.yaml` - GP3 EBS storage classes
- `alb-ingress.yaml` - ALB Ingress for OSMO service

## Common Options

All scripts support:

| Option | Description |
|--------|-------------|
| `-h, --help` | Show help message |
| `-t, --tf-dir` | Terraform directory (default: ../001-iac) |
| `--config-preview` | Print configuration without deploying |

## Cleanup

Uninstall in reverse order:

```bash
cd cleanup/

# Uninstall backend operator
./uninstall-osmo-backend.sh

# Uninstall control plane
./uninstall-osmo-control-plane.sh --delete-secrets

# Uninstall AWS prerequisites
./uninstall-aws-prerequisites.sh

# Finally, destroy infrastructure
cd ../../001-iac && terraform destroy
```

## Troubleshooting

### Check pod status
```bash
kubectl get pods -n osmo
kubectl describe pod <pod-name> -n osmo
kubectl logs <pod-name> -n osmo
```

### Pods in CrashLoopBackOff

In 6.3 the gateway (Envoy + oauth2-proxy) is a **standalone deployment** (`osmo-gateway-envoy`), not sidecars inside each service pod. The OSMO services (`osmo-service`, `osmo-worker`, `osmo-agent`, `osmo-logger`, `osmo-router`, `osmo-ui`) run their single app container.

```bash
# Service app logs
kubectl logs deploy/osmo-service -n osmo --tail=100

# Gateway (Envoy + oauth2-proxy live in the gateway deployment)
kubectl logs deploy/osmo-gateway-envoy -n osmo --tail=100

# Other services
kubectl logs deploy/osmo-worker -n osmo --tail=100
kubectl logs deploy/osmo-agent  -n osmo --tail=100
kubectl logs deploy/osmo-logger -n osmo --tail=100
kubectl logs deploy/osmo-router -n osmo --tail=100
```

Common causes:
- **osmo-service crash-loops with a config error** → malformed `services.configs.*`. In ConfigMap mode the loader fails fast; check `kubectl describe configmap osmo-service-configs -n osmo` for a `ConfigMapReloadFailed` event, fix the values, re-run `03`.
- **oauth2-proxy** missing/invalid `client_secret`/`cookie_secret` in secret `oauth2-proxy-secrets`, or (new in 6.3) it can't reach the Redis session store → verify `gateway.oauth2Proxy.redis.serviceName`.
- **service can't reach RDS/Redis** → check security groups and secrets.

Fix the secret/values/connectivity, then restart: `kubectl rollout restart deploy -n osmo`.

### Domain (https://osmo-aws...) shows nothing / not available

If Route 53 has the correct A/AAAA alias to the ALB but the site doesn’t load:

1. **Use HTTPS** – The deploy sets an ACM cert on the ingress; use `https://osmo-aws.example.com`. If you use `http://`, the ALB may redirect or only have an HTTPS listener.
2. **ALB target health** – In AWS Console: EC2 → Target Groups → select the target group for the ALB → Targets. Ensure targets are “Healthy”. If they are “Unhealthy”, fix pod readiness (see CrashLoopBackOff above) or health check path/codes in the ingress annotations.
3. **Pods and ingress** – All control-plane pods should be Ready and the ingress should have an ADDRESS:
   ```bash
   kubectl get pods -n osmo
   kubectl get ingress -n osmo
   ```
4. **Security groups** – The ALB security group (from Terraform output `alb_security_group_id`) must allow inbound 80/443 from the internet (or your IP if restricted); the EKS node security group must allow traffic from the ALB on the NodePort used by the service.

### Check Helm releases
```bash
helm list -n osmo
helm status osmo-service -n osmo
```

### Check Terraform outputs
```bash
cd ../001-iac && terraform output
```

See `../docs/troubleshooting.md` for more details.
