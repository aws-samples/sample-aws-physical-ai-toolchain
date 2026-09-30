#!/usr/bin/env bash
# SPDX-License-Identifier: MIT-0
#
# Event pre-provisioning for the "Cosmos 3 on EKS" lab.
#
# Creates the Cosmos 3 EKS cluster with this repo's Terraform, then stages the large
# artifacts, so workshop attendees don't spend their session waiting on downloads.
# Designed to run unattended in CodeBuild (see the workshop CloudFormation template),
# but it works from any shell with credentials.
#
# Two properties matter for a 100-account scale event:
#   * Idempotent - safe to re-run; Terraform reconciles and the staging Job is replaced.
#   * Observable  - fails loudly with a non-zero exit so a per-account health check
#                   (`aws codebuild batch-get-builds`) tells you which accounts are ready.
#
# Environment:
#   AWS_REGION_NAME     (required) Region to deploy into
#   STATE_BUCKET        (required) S3 bucket for Terraform remote state
#   ENVIRONMENT         Name prefix for created resources (default dev). Change it to
#                       isolate this deployment when the account already has a Cosmos 3
#                       cluster: the prefix appears in the cluster name AND in IAM role
#                       names that hardcode "cosmos3", so a different eks_cluster_name
#                       alone is NOT enough to avoid collisions.
#                       NOTE: the IRSA policy scopes S3 to
#                       physical-ai-<ENVIRONMENT>-datasets-<account>, so a non-default
#                       value needs a matching Foundation deployment for pods to read S3.
#   EKS_CLUSTER_NAME    Cluster name suffix               (default cosmos3)
#   GPU_INSTANCE_TYPE   GPU node instance type            (default g6e.4xlarge)
#   GPU_AZ              AZ to pin the GPU node group to   (default: any private subnet)
#   GPU_CR_ID           Capacity Block ID, if using one   (default: on-demand)
#   GPU_DESIRED_SIZE    GPU nodes to pre-warm             (default 1)
#   WORKSTATION_ROLE_ARN  IAM role granted EKS cluster-admin, so the workshop
#                       workstation can run kubectl and `terraform plan` (the Helm
#                       provider must reach the Kubernetes API to refresh state).
#   CENTRAL_WEIGHTS_S3  Central S3 prefix with Cosmos 3 Nano weights (optional)
#   COSMOS_IMAGE_URI    ECR image for vllm-omni cosmos3; also warms the node's image cache
#   TF_DIR              Terraform directory               (default: <repo>/cosmos-on-aws/infra)

set -euo pipefail

: "${AWS_REGION_NAME:?set AWS_REGION_NAME}"
: "${STATE_BUCKET:?set STATE_BUCKET}"
ENVIRONMENT="${ENVIRONMENT:-dev}"
EKS_CLUSTER_NAME="${EKS_CLUSTER_NAME:-cosmos3}"
GPU_INSTANCE_TYPE="${GPU_INSTANCE_TYPE:-g6e.4xlarge}"
GPU_AZ="${GPU_AZ:-}"
GPU_CR_ID="${GPU_CR_ID:-}"
GPU_DESIRED_SIZE="${GPU_DESIRED_SIZE:-1}"
CENTRAL_WEIGHTS_S3="${CENTRAL_WEIGHTS_S3:-}"
COSMOS_IMAGE_URI="${COSMOS_IMAGE_URI:-}"
WORKSTATION_ROLE_ARN="${WORKSTATION_ROLE_ARN:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF_DIR="${TF_DIR:-$SCRIPT_DIR/infra}"
TERRAFORM_VERSION="${TERRAFORM_VERSION:-1.16.4}"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

#-----------------------------------------------------------------------------------
# 1. Tooling
#-----------------------------------------------------------------------------------
if ! command -v terraform >/dev/null 2>&1; then
  log "Installing Terraform ${TERRAFORM_VERSION}"
  curl -fsSL -o /tmp/tf.zip \
    "https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_amd64.zip"
  unzip -q -o /tmp/tf.zip -d /usr/local/bin && rm -f /tmp/tf.zip
fi
if ! command -v kubectl >/dev/null 2>&1; then
  log "Installing kubectl"
  curl -fsSL -o /usr/local/bin/kubectl \
    "https://dl.k8s.io/release/$(curl -fsSL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
  chmod +x /usr/local/bin/kubectl
fi
terraform version | head -1

#-----------------------------------------------------------------------------------
# 2. Terraform: EKS cluster with a PRE-WARMED GPU node group
#
# The node group starts at desired=GPU_DESIRED_SIZE rather than 0 so the first job an
# attendee submits schedules immediately, instead of waiting ~5 min for Cluster
# Autoscaler to bring a node up (and another ~4 min to pull a 30 GB image).
#
# Remote state lives in STATE_BUCKET so attendees can later run
# `terraform init && terraform plan` from the workstation and see "No changes" -
# proof that the code in this repo is what built their cluster.
#-----------------------------------------------------------------------------------
cd "$TF_DIR"
cat > backend.tf <<EOF
terraform {
  backend "s3" {
    bucket = "${STATE_BUCKET}"
    key    = "cosmos-on-aws/eks/terraform.tfstate"
    region = "${AWS_REGION_NAME}"
  }
}
EOF
log "Wrote backend.tf -> s3://${STATE_BUCKET}/cosmos-on-aws/eks/terraform.tfstate"
log "Environment prefix: ${ENVIRONMENT}  Cluster suffix: ${EKS_CLUSTER_NAME}"

# Write the variable values to terraform.tfvars instead of passing them as -var flags.
# Terraform auto-loads this file, so an attendee can later run a bare
# `terraform init && terraform plan` and get "No changes" - with -var flags they would
# have to retype every value identically or the plan would show spurious drift.
# The file is also copied to the state bucket so the workstation can fetch the exact
# same values.
ADMIN_ARNS="[]"
if [ -n "$WORKSTATION_ROLE_ARN" ]; then
  ADMIN_ARNS="[\"${WORKSTATION_ROLE_ARN}\"]"
fi
cat > terraform.tfvars <<EOF
aws_region                      = "${AWS_REGION_NAME}"
environment                     = "${ENVIRONMENT}"
enable_eks_cluster              = true
enable_ec2_server               = false
eks_cluster_name                = "${EKS_CLUSTER_NAME}"
eks_gpu_node_instance_type      = "${GPU_INSTANCE_TYPE}"
eks_gpu_node_desired_size       = ${GPU_DESIRED_SIZE}
eks_gpu_availability_zone       = "${GPU_AZ}"
eks_gpu_capacity_reservation_id = "${GPU_CR_ID}"
# Grants the workstation IAM role EKS cluster-admin. Without an access entry the
# Helm provider cannot reach the Kubernetes API and even `terraform plan` fails with
# "Kubernetes cluster unreachable".
eks_admin_principal_arns        = ${ADMIN_ARNS}
EOF
log "Wrote terraform.tfvars"
cat terraform.tfvars

terraform init -input=false -upgrade
log "Applying Terraform (EKS cluster + GPU node group)"
terraform apply -input=false -auto-approve

# Publish the tfvars next to the state so the workstation uses identical values.
aws s3 cp terraform.tfvars "s3://${STATE_BUCKET}/cosmos-on-aws/eks/terraform.tfvars" \
  --only-show-errors || log "WARNING: could not publish terraform.tfvars"

CLUSTER="$(terraform output -raw eks_cluster_name)"
IRSA_ROLE="$(terraform output -raw eks_pod_irsa_role_arn)"
log "Cluster: ${CLUSTER}"

aws eks update-kubeconfig --name "$CLUSTER" --region "$AWS_REGION_NAME"
kubectl get nodes -o wide

#-----------------------------------------------------------------------------------
# 3. Service account bound to the Terraform-created IRSA role.
#
# Without this the staging (and generation) pods fall back to the default
# ServiceAccount, which resolves to the EKS node role - no S3 access at all.
#-----------------------------------------------------------------------------------
kubectl create serviceaccount cosmos3-generator -n default --dry-run=client -o yaml | kubectl apply -f -
kubectl annotate serviceaccount cosmos3-generator -n default \
  "eks.amazonaws.com/role-arn=${IRSA_ROLE}" --overwrite
log "ServiceAccount cosmos3-generator -> ${IRSA_ROLE}"

if [ -z "$CENTRAL_WEIGHTS_S3" ] && [ -z "$COSMOS_IMAGE_URI" ]; then
  log "No central artifacts configured (CENTRAL_WEIGHTS_S3 / COSMOS_IMAGE_URI) - skipping staging"
  log "Pre-provisioning complete (cluster only)"
  exit 0
fi

#-----------------------------------------------------------------------------------
# 4. Relay weights: central account -> this account's Foundation datasets bucket.
#
# The pod reads from the Foundation bucket rather than the central one because the
# IRSA policy Terraform creates already trusts it - no extra IAM required, and no
# participant ever needs Hugging Face credentials or a gated-model license.
#-----------------------------------------------------------------------------------
FOUNDATION_BUCKET="$(aws ssm get-parameter --name /physical-ai/datasets-bucket \
  --region "$AWS_REGION_NAME" --query Parameter.Value --output text)"
log "Foundation bucket: ${FOUNDATION_BUCKET}"

if [ -n "$CENTRAL_WEIGHTS_S3" ]; then
  log "Relaying weights from ${CENTRAL_WEIGHTS_S3}"
  aws s3 sync "$CENTRAL_WEIGHTS_S3" \
    "s3://${FOUNDATION_BUCKET}/cosmos3-assets/weights/" --only-show-errors
fi

#-----------------------------------------------------------------------------------
# 5. Staging Job: node-local weight cache + image pre-pull, in one step.
#
# hostPath, deliberately not EFS: eks.tf builds the cluster in the Foundation VPC
# (read from SSM), so the workshop stack's EFS is in a different VPC and unreachable;
# the cluster also installs ebs-csi but not efs-csi. The GPU node is pre-warmed and
# long-lived, so /opt/cosmos3 persists for every pod that lands on it. If the node is
# ever replaced, re-run this script (or just re-apply the Job).
#
# Running the real Cosmos image here is what warms the kubelet image cache, so the
# 30 GB pre-pull and the weight copy are the same operation.
#-----------------------------------------------------------------------------------
STAGE_IMAGE="${COSMOS_IMAGE_URI:-public.ecr.aws/aws-cli/aws-cli:latest}"
log "Staging with image ${STAGE_IMAGE}"

kubectl delete job cosmos3-staging -n default --ignore-not-found
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: cosmos3-staging
  namespace: default
spec:
  backoffLimit: 2
  template:
    spec:
      restartPolicy: Never
      serviceAccountName: cosmos3-generator
      nodeSelector:
        node.kubernetes.io/purpose: gpu
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Equal"
          value: "true"
          effect: "NoSchedule"
      containers:
        - name: stage
          image: ${STAGE_IMAGE}
          command: ["/bin/bash", "-lc"]
          args:
            - |
              set -euo pipefail
              command -v aws >/dev/null 2>&1 || pip install -q awscli
              mkdir -p /models/cosmos3-nano
              if [ -n "${CENTRAL_WEIGHTS_S3}" ]; then
                echo "Copying weights to node-local disk..."
                aws s3 sync "s3://${FOUNDATION_BUCKET}/cosmos3-assets/weights/" \
                  /models/cosmos3-nano/ --only-show-errors
                du -sh /models/cosmos3-nano
              fi
              nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
              echo STAGING_COMPLETE
          volumeMounts:
            - { name: models, mountPath: /models }
      volumes:
        - name: models
          hostPath:
            path: /opt/cosmos3
            type: DirectoryOrCreate
EOF

log "Waiting for staging to finish (weights + image pull can take ~10 min)"
if kubectl wait --for=condition=complete --timeout=45m job/cosmos3-staging -n default; then
  kubectl logs job/cosmos3-staging -n default --tail=10 || true
  log "Pre-provisioning complete - account is event-ready"
else
  log "ERROR: staging Job did not complete; recent logs:"
  kubectl logs job/cosmos3-staging -n default --tail=50 || true
  kubectl describe job cosmos3-staging -n default | tail -20 || true
  exit 1
fi
