# Teardown Guide

Complete, verified procedure for tearing down an OSMO-on-AWS deployment.

> The `## Cleanup` section in the main [README](../README.md) is a short summary.
> This document is the authoritative, step-by-step teardown reference including
> the ordering constraints and gotchas that a naive `terraform destroy` will hit.

## TL;DR

```bash
# 1. Remove in-cluster workloads (lets the LB Controller delete ALBs first)
cd 002-setup/cleanup
./uninstall-osmo-backend.sh        --delete-namespaces --force
./uninstall-osmo-control-plane.sh  --delete-namespace --delete-secrets --force
helm uninstall keycloak -n keycloak; kubectl delete namespace keycloak --wait=false
./uninstall-aws-prerequisites.sh   --force

# 2. WAIT for ALBs + ENIs to drain (see "Step 2" — this prevents a stuck destroy)

# 3. Destroy the infrastructure
cd ../../001-iac
terraform destroy
```

## Why ordering matters

The AWS Load Balancer Controller creates **ALBs, target groups, security groups,
and ENIs outside of Terraform** (in response to Kubernetes `Ingress`/`Service`
objects). Terraform does not know about these resources. If you run
`terraform destroy` while they still exist, the leftover ENIs keep the subnets
and VPC "in use", and **VPC/subnet deletion fails**.

The correct order is therefore:

1. Delete the Kubernetes workloads and ingresses → the LB Controller deletes the ALBs.
2. Uninstall the LB Controller **last** (while it can still clean up).
3. Wait for the ALBs/ENIs to actually disappear (deletion is asynchronous).
4. `terraform destroy`.

## Prerequisites

```bash
# Point kubectl at the cluster (values from terraform output)
aws eks update-kubeconfig --region <aws_region> --name <cluster_name>

# Handy: capture identifiers used below
cd 001-iac
REGION=$(terraform output -raw aws_region)
VPC_ID=$(terraform output -raw vpc_id)
echo "region=$REGION vpc=$VPC_ID"
```

## Step 1 — Uninstall in-cluster components

Run from `002-setup/cleanup/`:

```bash
./uninstall-osmo-backend.sh        --delete-namespaces --force
./uninstall-osmo-control-plane.sh  --delete-namespace --delete-secrets --force
helm uninstall keycloak -n keycloak; kubectl delete namespace keycloak --wait=false
./uninstall-aws-prerequisites.sh   --force
```

| Script | Removes |
|--------|---------|
| `uninstall-osmo-backend.sh` | `osmo-operator` release, operator token, (optionally) operator + workflows namespaces |
| `uninstall-osmo-control-plane.sh` | `service` release (API + router + UI + gateway), legacy `ui`/`router` releases, (optionally) secrets + namespace |
| `helm uninstall keycloak` | Keycloak release + its ingress/ALB |
| `uninstall-aws-prerequisites.sh` | GPU Operator, External Secrets, **AWS LB Controller**, storage classes |

> **Gap to be aware of:** `uninstall-aws-prerequisites.sh` does **not** remove
> `external-dns` or `aws-for-fluent-bit` (both installed by
> `01-deploy-aws-prerequisites.sh`). When you destroy the whole cluster they go
> away with it, but **`external-dns` leaves orphaned Route53 records** pointing
> at deleted ALBs. Clean them up manually afterward (see Step 4).

## Step 2 — Wait for ALBs and ENIs to drain (critical)

Poll until **both** commands return empty:

```bash
# No ALBs left in the VPC
aws elbv2 describe-load-balancers --region "$REGION" \
  --query "LoadBalancers[?VpcId=='$VPC_ID'].LoadBalancerName" --output text

# No in-use ENIs left in the VPC
aws ec2 describe-network-interfaces --region "$REGION" \
  --filters "Name=vpc-id,Values=$VPC_ID" \
  --query "NetworkInterfaces[?Status=='in-use'].[NetworkInterfaceId,Description]" \
  --output text
```

Only proceed once these are empty (typically 1–3 minutes after Step 1).

## Step 3 — `terraform destroy`

```bash
cd 001-iac
terraform destroy   # review the plan, then confirm
```

What to expect for this repo's defaults:

- **RDS deletion protection** must be `false` (`rds_deletion_protection` in
  `terraform.tfvars`) or destroy will refuse to delete the database.
- **S3 buckets** are emptied + deleted only when `s3_force_destroy = true`.
  Otherwise destroy fails on non-empty (versioned) buckets.
- **Secrets Manager** entries use `secrets_recovery_window_days`. With `0` they
  are deleted immediately (no name collision on recreate); with `>0` they enter
  a scheduled-deletion window and the same names cannot be reused until purged.
- **Final RDS snapshot:** when `environment = "prod"`, `skip_final_snapshot` is
  `false`, so destroy creates a final snapshot (auto-named with a random suffix
  — no collision, but it persists and costs a little). Delete it manually if not
  wanted.
- **KMS key** is scheduled for deletion (7–30 day window), not removed instantly;
  its alias is freed immediately so a fresh deploy can recreate it.

## Step 4 — Post-destroy cleanup (manual)

```bash
# Orphaned Route53 records left by external-dns (point at deleted ALBs)
aws route53 list-resource-record-sets --hosted-zone-id <route53_zone_id> \
  --query "ResourceRecordSets[?Type=='A' || Type=='CNAME'].Name"
# delete any osmo-* records that remain

# Leftover RDS final snapshot (prod only)
aws rds describe-db-snapshots --region "$REGION" --snapshot-type manual \
  --query "DBSnapshots[?contains(DBSnapshotIdentifier,'osmo')].DBSnapshotIdentifier"

# CloudWatch log group (if CloudWatch logging was enabled)
aws logs describe-log-groups --region "$REGION" \
  --log-group-name-prefix "/aws/eks/<cluster_name>"
```

## Quick reference: what is / isn't managed by Terraform

| Created by Terraform (`terraform destroy` removes) | Created in-cluster (must uninstall first) |
|----------------------------------------------------|-------------------------------------------|
| VPC, subnets, NAT/IGW, route tables, VPC endpoints | ALBs / target groups (via LB Controller)  |
| EKS cluster + node groups, IRSA roles              | OSMO `service` / backend-operator releases |
| RDS, ElastiCache, S3, KMS, Secrets Manager         | Keycloak release                          |
| ACM certs, WAF, AMP workspace, CloudWatch groups   | GPU Operator, External Secrets, external-dns, Fluent Bit |
