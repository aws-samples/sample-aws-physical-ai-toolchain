# =============================================================================
# EKS: COSMOS 3 GENERATION CLUSTER (dedicated, minimal — not shared with OSMO)
# =============================================================================
#
# A standalone EKS cluster scoped to running the Cosmos3-Super vLLM-Omni server
# as a Kubernetes Job. Two node groups:
#   - system: small on-demand nodes for CoreDNS/addons (always on)
#   - gpu:    p5.48xlarge / p5en.48xlarge, desired_size=0 by default — scale to
#             1 only when submitting a generation Job, same "pay only while
#             generating" model as the EC2 path.
#
# GPU node group targets an EC2 Capacity Block via a launch template, the same
# mechanism used in ec2.tf. Purchase the block separately (see Step 1 of
# ec2-deployment-guide.md) — Capacity Block purchase is not Terraform-managed
# because it activates asynchronously outside a single `apply` lifecycle.
#
# Toggle with var.enable_eks_cluster (default false) so `terraform apply` is
# safe to run without standing up a cluster (control plane bills ~$0.10/hr
# once created, regardless of node count).

variable "enable_eks_cluster" {
  description = "Deploy the dedicated Cosmos 3 EKS cluster. Cluster control plane bills continuously once created."
  type        = bool
  default     = false
}

variable "eks_cluster_name" {
  description = "Name for the Cosmos 3 EKS cluster"
  type        = string
  default     = "cosmos3"
}

variable "eks_kubernetes_version" {
  description = "Kubernetes version for the EKS cluster"
  type        = string
  default     = "1.31"
}

variable "eks_admin_principal_arns" {
  description = "Additional IAM principal ARNs granted EKS cluster admin access (beyond the applying identity)"
  type        = list(string)
  default     = []
}

variable "eks_system_node_instance_types" {
  description = "Instance types for the always-on system node group"
  type        = list(string)
  default     = ["t3.medium"]
}

variable "eks_gpu_node_instance_type" {
  description = "Instance type for the GPU node group (must match the Capacity Block's reserved type)"
  type        = string
  default     = "p5.48xlarge"
}

variable "eks_gpu_node_desired_size" {
  description = "Initial desired GPU nodes at node-group creation. After that, Cluster Autoscaler manages this automatically (scale up on a pending GPU pod, scale down ~10 min after the pod completes) — see the cluster-autoscaler Helm release below. Leave at 0; only used for the very first apply."
  type        = number
  default     = 0
}

variable "eks_cluster_autoscaler_version" {
  description = "Helm chart version for cluster-autoscaler (defaults to latest tested against eks_kubernetes_version 1.31)"
  type        = string
  default     = "9.46.6"
}

variable "eks_gpu_node_volume_size_gb" {
  description = "Root EBS volume size for GPU nodes (container image + model weight cache)"
  type        = number
  default     = 500
}

variable "eks_gpu_capacity_reservation_id" {
  description = "ID of an ACTIVE EC2 Capacity Block for the GPU node group to target. Leave empty to launch on-demand (will likely fail — P5 capacity is scarce)."
  type        = string
  default     = ""
}

variable "eks_gpu_availability_zone" {
  description = "AZ of the Capacity Block. The GPU node group's subnet is pinned to this AZ — EKS node groups otherwise span all private subnets/AZs, which fails Capacity Block launches with an AZ mismatch (the reservation is tied to one specific AZ)."
  type        = string
  default     = ""
}

data "aws_ssm_parameter" "eks_vpc_id" {
  count = var.enable_eks_cluster ? 1 : 0
  name  = "/${var.project_name}/vpc-id"
}

data "aws_ssm_parameter" "eks_private_subnet_ids" {
  count = var.enable_eks_cluster ? 1 : 0
  name  = "/${var.project_name}/private-subnet-ids"
}

data "aws_caller_identity" "eks_current" {
  count = var.enable_eks_cluster ? 1 : 0
}

# GPU node group must be pinned to the single AZ the Capacity Block is
# reserved in — EKS picks a subnet across whatever's passed to subnet_ids,
# and a mismatch fails node creation with an AZ-mismatch error from EC2.
data "aws_subnet" "eks_gpu_az_subnet" {
  count = var.enable_eks_cluster && var.eks_gpu_availability_zone != "" ? 1 : 0
  filter {
    name   = "subnet-id"
    values = split(",", data.aws_ssm_parameter.eks_private_subnet_ids[0].value)
  }
  availability_zone = var.eks_gpu_availability_zone
}

locals {
  eks_private_subnet_ids = var.enable_eks_cluster ? split(",", data.aws_ssm_parameter.eks_private_subnet_ids[0].value) : []

  # GPU node group subnet: pinned to the Capacity Block's AZ when specified,
  # otherwise falls back to all private subnets (on-demand path).
  eks_gpu_subnet_ids = var.enable_eks_cluster && var.eks_gpu_availability_zone != "" ? [data.aws_subnet.eks_gpu_az_subnet[0].id] : local.eks_private_subnet_ids

  # Launch template for the GPU node group only needs to add Capacity Block
  # targeting; everything else (AMI, IAM, security groups) comes from the
  # EKS managed node group defaults.
  eks_gpu_capacity_block_specified = var.eks_gpu_capacity_reservation_id != ""
}

# =============================================================================
# EKS CLUSTER + NODE GROUPS
# =============================================================================

module "cosmos3_eks" {
  count   = var.enable_eks_cluster ? 1 : 0
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = "${local.prefix}-${var.eks_cluster_name}"
  cluster_version = var.eks_kubernetes_version

  vpc_id     = data.aws_ssm_parameter.eks_vpc_id[0].value
  subnet_ids = local.eks_private_subnet_ids

  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true

  enable_irsa                              = true
  enable_cluster_creator_admin_permissions = true

  access_entries = {
    for idx, principal_arn in var.eks_admin_principal_arns : "admin-${idx}" => {
      principal_arn = principal_arn
      policy_associations = {
        admin = {
          policy_arn   = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"
          access_scope = { type = "cluster" }
        }
      }
    }
  }

  cluster_addons = {
    coredns    = { most_recent = true }
    kube-proxy = { most_recent = true }
    vpc-cni    = { most_recent = true, before_compute = true }
    aws-ebs-csi-driver = {
      most_recent              = true
      service_account_role_arn = module.ebs_csi_irsa[0].iam_role_arn
    }
  }

  eks_managed_node_groups = {
    system = {
      name           = "cosmos3-system"
      instance_types = var.eks_system_node_instance_types
      capacity_type  = "ON_DEMAND"
      min_size       = 1
      max_size       = 2
      desired_size   = 1
      subnet_ids     = local.eks_private_subnet_ids

      labels = { "node.kubernetes.io/purpose" = "system" }
    }

    gpu = {
      name           = "cosmos3-gpu"
      instance_types = [var.eks_gpu_node_instance_type]
      # Capacity Block nodes must use CAPACITY_BLOCK capacity type.
      capacity_type = local.eks_gpu_capacity_block_specified ? "CAPACITY_BLOCK" : "ON_DEMAND"
      ami_type      = "AL2_x86_64_GPU"

      min_size     = 0
      max_size     = 1
      desired_size = var.eks_gpu_node_desired_size
      subnet_ids   = local.eks_gpu_subnet_ids

      labels = {
        "node.kubernetes.io/purpose" = "gpu"
        "nvidia.com/gpu.present"     = "true"
      }

      taints = {
        gpu = {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }

      # Cluster Autoscaler tags on the underlying ASG. These let autoscaler
      # discover and scale this node group FROM ZERO — when scaled to 0 there
      # is no live node for it to inspect labels/taints/GPU count from, so it
      # reads them from these node-template tags instead. Without the
      # resources/nvidia.com/gpu tag specifically, autoscaler can't tell a
      # pending pod's `nvidia.com/gpu: 8` request would fit a node it hasn't
      # seen yet, and will never scale up.
      tags = {
        "k8s.io/cluster-autoscaler/enabled"                                        = "true"
        "k8s.io/cluster-autoscaler/${local.prefix}-${var.eks_cluster_name}"        = "owned"
        "k8s.io/cluster-autoscaler/node-template/label/node.kubernetes.io/purpose" = "gpu"
        "k8s.io/cluster-autoscaler/node-template/label/nvidia.com/gpu.present"     = "true"
        "k8s.io/cluster-autoscaler/node-template/taint/nvidia.com/gpu"             = "true:NoSchedule"
        "k8s.io/cluster-autoscaler/node-template/resources/nvidia.com/gpu"         = "8"
      }

      # Capacity Block targeting — same mechanism as the EC2 instance
      # (capacity_reservation_specification), applied via the node group's
      # generated launch template. Must be `{}` (not null) when unset — the
      # module checks length() on this value.
      capacity_reservation_specification = local.eks_gpu_capacity_block_specified ? {
        capacity_reservation_target = {
          capacity_reservation_id = var.eks_gpu_capacity_reservation_id
        }
      } : {}

      # Capacity Block launches ALSO require instance_market_options with
      # market_type="capacity-block" on the launch template (separate from
      # capacity_reservation_specification above) — same requirement we hit
      # launching the standalone EC2 instance in ec2.tf. Without this the API
      # rejects the node group with "market type (purchasing) option is not
      # valid" even though capacity_reservation_specification is set.
      instance_market_options = local.eks_gpu_capacity_block_specified ? {
        market_type = "capacity-block"
      } : {}

      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = var.eks_gpu_node_volume_size_gb
            volume_type           = "gp3"
            encrypted             = true
            delete_on_termination = true
          }
        }
      }
    }
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "cosmos3-eks"
  }
}

# =============================================================================
# EBS CSI DRIVER IRSA (required for the aws-ebs-csi-driver addon above)
# =============================================================================

module "ebs_csi_irsa" {
  count   = var.enable_eks_cluster ? 1 : 0
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name             = "${local.prefix}-cosmos3-ebs-csi-irsa"
  attach_ebs_csi_policy = true

  oidc_providers = {
    main = {
      # Referencing the cluster module directly here would create a circular
      # dependency (module needs this role ARN in cluster_addons, this role
      # needs the module's OIDC provider). Terraform resolves this because
      # both are in the same apply graph and the provider ARN is a stable,
      # computable attribute — but if you see a cycle error, remove the
      # aws-ebs-csi-driver entry from cluster_addons on first apply, then add
      # it back on a second apply once the OIDC provider exists.
      provider_arn               = module.cosmos3_eks[0].oidc_provider_arn
      namespace_service_accounts = ["kube-system:ebs-csi-controller-sa"]
    }
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "cosmos3-eks"
  }
}

# =============================================================================
# IRSA: COSMOS3 GENERATION POD (S3 write + Secrets Manager read)
# =============================================================================

resource "aws_iam_policy" "cosmos3_pod_s3" {
  count       = var.enable_eks_cluster ? 1 : 0
  name        = "${local.prefix}-cosmos3-pod-s3"
  description = "S3 read/write for the Cosmos3 generation pod (cosmos-samples/ output, dataset input)"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ListDatasetsBucket"
        Effect = "Allow"
        Action = ["s3:ListBucket"]
        Resource = [
          "arn:aws:s3:::${var.project_name}-${var.environment}-datasets-${data.aws_caller_identity.eks_current[0].account_id}"
        ]
      },
      {
        Sid    = "ReadWriteDatasetsObjects"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:PutObject"]
        Resource = [
          "arn:aws:s3:::${var.project_name}-${var.environment}-datasets-${data.aws_caller_identity.eks_current[0].account_id}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_policy" "cosmos3_pod_secrets" {
  count       = var.enable_eks_cluster ? 1 : 0
  name        = "${local.prefix}-cosmos3-pod-secrets"
  description = "Secrets Manager read for the Cosmos3 generation pod (HF token)"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid      = "ReadHfToken"
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = ["arn:aws:secretsmanager:${local.region}:${data.aws_caller_identity.eks_current[0].account_id}:secret:${var.project_name}/hf-token*"]
    }]
  })
}

module "cosmos3_pod_irsa" {
  count   = var.enable_eks_cluster ? 1 : 0
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "${local.prefix}-cosmos3-pod-irsa"

  role_policy_arns = {
    s3      = aws_iam_policy.cosmos3_pod_s3[0].arn
    secrets = aws_iam_policy.cosmos3_pod_secrets[0].arn
  }

  oidc_providers = {
    main = {
      provider_arn               = module.cosmos3_eks[0].oidc_provider_arn
      namespace_service_accounts = ["default:cosmos3-generator"]
    }
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "cosmos3-eks"
  }
}

# =============================================================================
# CLUSTER AUTOSCALER (scale the GPU node group 0 -> 1 -> 0 automatically)
# =============================================================================
#
# Watches for pods that can't schedule because there's no node with capacity
# (e.g. the cosmos3-vllm-omni Job requesting nvidia.com/gpu: 8 while the GPU
# node group sits at desired=0), scales the ASG up, waits for the pod to
# schedule and run, then scales back to 0 ~10 min after the node goes idle.
# This replaces manually running `aws eks update-nodegroup-config` before/
# after each generation Job.

module "cluster_autoscaler_irsa" {
  count   = var.enable_eks_cluster ? 1 : 0
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name                        = "${local.prefix}-cosmos3-cluster-autoscaler-irsa"
  attach_cluster_autoscaler_policy = true
  cluster_autoscaler_cluster_names = [module.cosmos3_eks[0].cluster_name]

  oidc_providers = {
    main = {
      provider_arn               = module.cosmos3_eks[0].oidc_provider_arn
      namespace_service_accounts = ["kube-system:cluster-autoscaler"]
    }
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "cosmos3-eks"
  }
}

resource "helm_release" "cluster_autoscaler" {
  count      = var.enable_eks_cluster ? 1 : 0
  name       = "cluster-autoscaler"
  repository = "https://kubernetes.github.io/autoscaler"
  chart      = "cluster-autoscaler"
  version    = var.eks_cluster_autoscaler_version
  namespace  = "kube-system"

  set {
    name  = "autoDiscovery.clusterName"
    value = module.cosmos3_eks[0].cluster_name
  }
  set {
    name  = "awsRegion"
    value = var.aws_region
  }
  set {
    name  = "rbac.serviceAccount.name"
    value = "cluster-autoscaler"
  }
  set {
    name  = "rbac.serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn"
    value = module.cluster_autoscaler_irsa[0].iam_role_arn
  }
  set {
    # Default is 10 min; matches the "generate then idle" pattern here
    # closely enough without being so aggressive it tears down a node
    # between back-to-back generation requests.
    name  = "extraArgs.scale-down-unneeded-time"
    value = "10m"
  }
  set {
    # Required so autoscaler evaluates a scaled-to-zero node group's
    # node-template tags (set on the GPU node group above) instead of
    # skipping it for having no nodes to inspect.
    name  = "extraArgs.balance-similar-node-groups"
    value = "false"
  }

  depends_on = [module.cosmos3_eks]
}

# =============================================================================
# OUTPUTS
# =============================================================================

output "eks_cluster_name" {
  description = "Cosmos 3 EKS cluster name (if enabled)"
  value       = var.enable_eks_cluster ? module.cosmos3_eks[0].cluster_name : null
}

output "eks_cluster_endpoint" {
  description = "Cosmos 3 EKS cluster API endpoint (if enabled)"
  value       = var.enable_eks_cluster ? module.cosmos3_eks[0].cluster_endpoint : null
}

output "eks_pod_irsa_role_arn" {
  description = "IRSA role ARN for the cosmos3-generator service account (if enabled)"
  value       = var.enable_eks_cluster ? module.cosmos3_pod_irsa[0].iam_role_arn : null
}
