# OSMO on AWS - Troubleshooting Guide

## Common Issues

### 1. Terraform Errors

#### "Error: creating EKS Cluster"

**Symptoms:**
```
Error: creating EKS Cluster (osmo-cluster): operation error EKS: CreateCluster
```

**Causes & Solutions:**
- IAM permissions insufficient → Ensure IAM user/role has EKS permissions
- Subnet configuration invalid → Check VPC CIDR ranges don't overlap
- Service quota exceeded → Request quota increase for EKS clusters

#### "Error: Provider produced inconsistent final plan"

**Solution:** Run `terraform refresh` then `terraform plan` again.

### 2. EKS Connection Issues

#### "Unable to connect to the server"

**Symptoms:**
```
Unable to connect to the server: dial tcp: lookup ... on ...: no such host
```

**Causes & Solutions:**
1. Kubeconfig not configured:
   ```bash
   aws eks update-kubeconfig --region <region> --name <cluster-name>
   ```

2. VPN required (private endpoint only):
   - Connect to VPN if cluster has `cluster_endpoint_public_access = false`

3. IAM permissions:
   - Ensure your IAM identity is in the cluster's access entries
   ```bash
   aws eks list-access-entries --cluster-name <cluster-name>
   ```

### 3. Pod Issues

#### Pods stuck in "Pending"

**Diagnosis:**
```bash
kubectl describe pod <pod-name> -n osmo
```

**Common causes:**

1. **Insufficient resources:**
   ```
   0/3 nodes are available: 3 Insufficient cpu
   ```
   → Scale up node group or reduce resource requests

2. **Node selector/taint issues:**
   ```
   0/3 nodes are available: 3 node(s) didn't match Pod's node affinity
   ```
   → Check node labels and pod tolerations

3. **PVC pending:**
   ```
   pod has unbound immediate PersistentVolumeClaims
   ```
   → Check storage class and EBS CSI driver

#### Pods stuck in "ImagePullBackOff"

**Diagnosis:**
```bash
kubectl describe pod <pod-name> -n osmo | grep -A5 "Events:"
```

**Solutions:**
1. NGC registry secret missing:
   ```bash
   kubectl create secret docker-registry ngc-registry-secret \
     -n osmo \
     --docker-server=nvcr.io \
     --docker-username='$oauthtoken' \
     --docker-password="<NGC_API_KEY>"
   ```

2. ECR authentication:
   ```bash
   # IRSA should handle this, but verify the role
   kubectl describe serviceaccount osmo-service -n osmo
   ```

#### Pods in "CrashLoopBackOff"

**Diagnosis:**
```bash
kubectl logs <pod-name> -n osmo --previous
```

**Common causes:**

1. **Database connection failed:**
   - Check RDS security group allows traffic from EKS
   - Verify database credentials in secret

2. **Redis connection failed:**
   - Check ElastiCache security group
   - Verify auth token in secret

3. **Missing configuration:**
   - Check ConfigMaps and Secrets are created
   - Verify environment variables

### 4. Database Connectivity

#### Cannot connect to RDS

**Test from a pod:**
```bash
kubectl run -it --rm psql-test --image=postgres:15 \
  --restart=Never -- \
  psql -h <rds-endpoint> -U osmo_admin -d osmo
```

**Checklist:**
- [ ] RDS security group allows port 5432 from EKS node security group
- [ ] Pod is in a private subnet that can reach RDS
- [ ] Password secret is correct

### 5. Redis Connectivity

#### Cannot connect to ElastiCache

**Test from a pod:**
```bash
kubectl run -it --rm redis-test --image=redis:7 \
  --restart=Never -- \
  redis-cli -h <redis-endpoint> -p 6379 --tls -a <auth-token> ping
```

**Checklist:**
- [ ] ElastiCache security group allows port 6379 from EKS
- [ ] TLS is enabled (required for auth tokens)
- [ ] Auth token secret is correct

### 6. Load Balancer Issues

#### ALB not provisioning

**Diagnosis:**
```bash
kubectl describe ingress osmo-service-ingress -n osmo
kubectl logs -n kube-system -l app.kubernetes.io/name=aws-load-balancer-controller
```

**Common causes:**

1. **Subnet tags missing:**
   - Public subnets need: `kubernetes.io/role/elb=1`
   - Private subnets need: `kubernetes.io/role/internal-elb=1`

2. **IAM permissions:**
   - Check AWS LB Controller IRSA role has correct policies

3. **Ingress class mismatch:**
   - Verify `ingressClassName: alb` in Ingress resource

### 7. GPU Issues

#### GPUs not detected

**Diagnosis:**
```bash
# Check GPU operator
kubectl get pods -n gpu-operator

# Check node labels
kubectl get nodes -l nvidia.com/gpu.present=true

# Check device plugin
kubectl logs -n gpu-operator -l app=nvidia-device-plugin-daemonset
```

**Solutions:**

1. **Driver not loaded:**
   - EKS GPU AMI should have drivers pre-installed
   - Check AMI type: `AL2_x86_64_GPU`

2. **Device plugin not running:**
   - Check GPU operator deployment
   - Verify node taints and tolerations

3. **Wrong instance type:**
   - Ensure nodes are GPU instance types (g4dn, g5, p3, p4d, etc.)

#### GPU Operator pods crash on containerd 2.x AMIs

**Symptom:** After a fresh deploy, pods in `gpu-operator` (notably
`nvidia-container-toolkit-daemonset` and `nvidia-operator-validator`) crash-loop,
and GPU nodes never advertise `nvidia.com/gpu` capacity.

**Cause:** GPU Operator **v24.9.0** ships a container-toolkit that is incompatible
with the **containerd 2.x** runtime on current EKS GPU AMIs — the toolkit writes a
`config.toml` in a schema that containerd 2.x rejects, so the runtime never gets
NVIDIA support wired in.

**Solution:** use GPU Operator **v25.10.1** or later (now the default in
`config/deployment-config.yaml.example` and `defaults.conf`). If you overrode it,
set `dependencies.gpu_operator_version: "v25.10.1"` (or `export
GPU_OPERATOR_VERSION=v25.10.1`) and re-run `02-deploy-gpu-infrastructure.sh`.

#### GPU nodes fail to launch — `VcpuLimitExceeded` / `InsufficientInstanceCapacity`

**Symptom:** the GPU node group never reaches its desired count; the backing EC2
Auto Scaling group shows failed scaling activities.

**Diagnosis:**
```bash
# Inspect the most recent scaling activities for the GPU node group's ASG
aws autoscaling describe-scaling-activities \
  --auto-scaling-group-name <gpu-node-group-asg> --max-items 5
```

**Causes & solutions:**
- **`VcpuLimitExceeded`** — the account's On-Demand GPU vCPU quota is too low. For
  the g5/g6 families the quota is **L-DB2E81BA** ("Running On-Demand G and VT
  instances"); the P family (p4d/p5) has a separate quota code. Request an increase
  in Service Quotas for the deployment region, then retry.
- **`InsufficientInstanceCapacity`** — AWS has no capacity for that instance type in
  the chosen AZ. Try another AZ or instance type, or back the node group with an
  On-Demand Capacity Reservation.

### 8. OSMO-Specific Issues

#### Backend not connecting to service

**Diagnosis:**
```bash
kubectl logs -n osmo -l app.kubernetes.io/name=osmo-backend-listener
```

**Checklist:**
- [ ] Service URL is correct and reachable
- [ ] Backend operator token is valid
- [ ] WebSocket connection is not blocked by security groups

#### Workflow stuck in "Pending"

**Diagnosis:**
```bash
kubectl get pods -n osmo-workflows
kubectl describe pod <workflow-pod> -n osmo-workflows
```

**Common causes:**
- GPU nodes not available (check autoscaler)
- Resource quota exceeded
- PVC cannot be provisioned
- GPU node root disk too small for large images (Isaac Sim / Cosmos) → see `gpu_node_volume_size` (default 500 GiB) in `terraform.tfvars`; symptom is `Evicted: node was low on resource: ephemeral-storage` or `ErrImagePull: no space left on device`.

#### Dataset write fails with S3 `AccessDenied` (`GetBucketLocation`)

**Symptom:** a workflow task fails during "Validating WRITE access for dataset output" with:
```
User: arn:aws:sts::<acct>:assumed-role/<...>-gpu-eks-node-group/<i-...> is not authorized
to perform: s3:GetBucketLocation ...
```

**Cause:** `osmo-ctrl` in the task pod fell back to the **EKS node IAM role** instead of using IRSA. In the 6.3 model, workflow task pods must run under the `osmo-workflow` ServiceAccount (annotated with the `osmo_workflow_role_arn`).

**Solutions:**
1. Verify the SA exists and is annotated:
   ```bash
   kubectl get sa osmo-workflow -n osmo-workflows -o yaml | grep role-arn
   ```
   If missing, re-run `05-deploy-osmo-backend.sh` (it creates + annotates the SA).
2. Verify the pod template pins the SA — `services.configs.podTemplates.gpu_tolerations.spec.serviceAccountName: osmo-workflow` in `values/osmo-control-plane.yaml`. Re-run `04` after changes.
3. Confirm the IRSA trust policy subject matches `osmo-workflows:osmo-workflow` (see `001-iac/modules/eks/irsa.tf`).
4. Ensure the `osmo-workflows` namespace allows HTTPS egress to STS + S3 (NetworkPolicy `allow-workflow-aws-egress`).

#### `s3://` workflow I/O fails with `ValueError: Invalid endpoint:`

**Symptom:** a workflow that reads/writes `s3://…` dataset or artifact paths fails
early with an error resembling:
```
ValueError: Invalid endpoint: https://s3..amazonaws.com
```
(note the empty region between the dots).

**Cause:** upstream bug NVIDIA/OSMO#1067 — on the IRSA credential path `osmo-ctrl`
constructs the S3 endpoint without a region, producing an invalid host. It surfaces
only when the task authenticates via **IRSA**, which is the required model on AWS
(company policy prohibits long-lived data-bucket credentials).

**Solution (workaround):** set `OSMO_SKIP_DATA_AUTH=1` on the task pod template.
This repo already applies it in `values/osmo-control-plane.yaml` under
`services.configs.podTemplates.default_ctrl`; if you customized pod templates,
re-add the env var and re-run `04-deploy-osmo-control-plane.sh`. Remove the
workaround once #1067 is fixed upstream.

#### Config change via CLI/API returns HTTP 409

**Cause:** 6.3 runs in **ConfigMap configuration mode** (`services.configs.enabled: true`), which makes the config API read-only.

**Solution:** edit the declarative values in `values/osmo-control-plane.yaml` (`services.configs.*`) and re-run `04-deploy-osmo-control-plane.sh`. A malformed config crash-loops new pods — check `kubectl describe configmap osmo-service-configs -n osmo` for a `ConfigMapReloadFailed` event.

#### `kubectl logs`/`exec`/`port-forward` fail with "Authorization error ... nodes/proxy"

**Cause:** the API server's kubelet client lacks RBAC for the `nodes/proxy` subresource (cluster-wide), which breaks logs/exec/port-forward — and therefore the bootstrap scripts that port-forward to `osmo-service`.

**Solution:** this is a cluster-level kubelet authorization issue (often from a custom node AMI bootstrapping kubelet authz differently). Verify a healthy fresh EKS cluster has the default apiserver→kubelet RBAC; if using a custom GPU AMI, ensure its bootstrap configures `--authorization-mode=Webhook` / `--authentication-token-webhook` consistently with the EKS-optimized system nodes.

### 9. Authentication / Login Issues

#### Forced to re-login roughly every hour

**Cause:** the `osmo` realm is created (by `03-deploy-keycloak.sh`) with short
lifetimes — `accessTokenLifespan` 3600s (1h) and `ssoSessionIdleTimeout` 1800s
(30m idle). When the session expires, oauth2-proxy bounces the browser back to
Keycloak for re-authentication.

**Solution:** raise the realm's `accessTokenLifespan` / `ssoSessionIdleTimeout`
(Keycloak Admin → Realm settings → Tokens/Sessions) to match your security posture,
then have users re-authenticate once. No OSMO redeploy is needed. To change the
baked-in defaults for future deploys, edit the realm-creation body in
`03-deploy-keycloak.sh`.

#### `osmo` CLI returns 401 / token expired

**Causes & solutions:**
- **Expired credential** — re-run `osmo login` to refresh. The device-login code
  itself is short-lived (`oauth2DeviceCodeLifespan` 600s / 10m), so complete the
  browser step promptly.
- **CLI / API version skew** — a CLI built against a different OSMO API version can
  fail auth or reject responses. Match the CLI to the deployed control plane
  (`osmo.version` in `config/deployment-config.yaml`, currently `6.3.1`).

## Diagnostic Commands

### Cluster Health
```bash
# Node status
kubectl get nodes -o wide

# All pods in osmo namespace
kubectl get pods -n osmo -o wide

# Events
kubectl get events -n osmo --sort-by='.lastTimestamp'
```

### Terraform State
```bash
cd 001-iac
terraform state list
terraform output
```

### AWS Resources
```bash
# EKS cluster
aws eks describe-cluster --name <cluster-name>

# Node groups
aws eks list-nodegroups --cluster-name <cluster-name>

# RDS
aws rds describe-db-instances --db-instance-identifier <rds-id>

# ElastiCache
aws elasticache describe-replication-groups --replication-group-id <redis-id>
```

### Helm Releases
```bash
helm list -A
helm history osmo-service -n osmo
helm get values osmo-service -n osmo
```

## Getting Help

1. **Check logs:** Always start with pod logs
2. **Review events:** `kubectl get events -n osmo`
3. **Terraform outputs:** Verify infrastructure is correctly deployed
4. **Security groups:** Common cause of connectivity issues
5. **IRSA roles:** Verify service account annotations
