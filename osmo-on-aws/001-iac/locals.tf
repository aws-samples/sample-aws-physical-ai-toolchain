# SPDX-License-Identifier: Apache-2.0

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

locals {
  # Naming convention
  name_prefix = "${var.cluster_name}-${var.environment}-${var.resource_suffix}"

  # Common tags for all resources
  common_tags = {
    Environment    = var.environment
    Project        = var.project_name
    Owner          = var.owner
    ManagedBy      = "terraform"
    ClusterName    = var.cluster_name
    ProjectName    = "OSMO on AWS"
    CodeRepository = "https://github.com/your-org/osmo-on-aws"
  }

  # Availability zones
  azs = slice(data.aws_availability_zones.available.names, 0, var.availability_zones_count)

  # VPC CIDR calculations
  # Private subnets: 10.0.0.0/19, 10.0.32.0/19, 10.0.64.0/19
  # Public subnets: 10.0.96.0/24, 10.0.97.0/24, 10.0.98.0/24
  # Database subnets: 10.0.100.0/24, 10.0.101.0/24, 10.0.102.0/24
  # ElastiCache subnets: 10.0.104.0/24, 10.0.105.0/24, 10.0.106.0/24
  private_subnets     = [for i in range(var.availability_zones_count) : cidrsubnet(var.vpc_cidr, 3, i)]
  public_subnets      = [for i in range(var.availability_zones_count) : cidrsubnet(var.vpc_cidr, 8, 96 + i)]
  database_subnets    = [for i in range(var.availability_zones_count) : cidrsubnet(var.vpc_cidr, 8, 100 + i)]
  elasticache_subnets = [for i in range(var.availability_zones_count) : cidrsubnet(var.vpc_cidr, 8, 104 + i)]

  # Deployment mode conditionals
  is_control_plane_only = var.deployment_mode == "control-plane-only"
  is_backend_only       = var.deployment_mode == "backend-only"

  # Component deployment flags based on mode
  deploy_postgresql = var.should_deploy_postgresql && !local.is_backend_only
  deploy_redis      = var.should_deploy_redis && !local.is_backend_only
  deploy_gpu_nodes  = var.should_deploy_gpu_nodes && !local.is_control_plane_only

  # S3 bucket names (account ID suffix ensures global uniqueness)
  s3_workflows_bucket = "${local.name_prefix}-workflows-${data.aws_caller_identity.current.account_id}"
  s3_datasets_bucket  = "${local.name_prefix}-datasets-${data.aws_caller_identity.current.account_id}"

  # Hostnames derived from Route53 zone
  osmo_hostname      = "osmo-aws.${var.route53_zone_name}"
  osmo_auth_hostname = "osmo-aws-auth.${var.route53_zone_name}"

  # IRSA service accounts
  osmo_namespace       = "osmo"
  osmo_service_account = "osmo-service"
  osmo_backend_sa      = "osmo-backend-operator"
  # Backend-operator chart deploys in the operator namespace and names its SAs
  # "<helm-release>-backend-{listener,worker}". The control-plane release is
  # "osmo-operator" (see 05-deploy-osmo-backend.sh).
  osmo_operator_namespace  = "osmo-operator"
  osmo_backend_listener_sa = "osmo-operator-backend-listener"
  osmo_backend_worker_sa   = "osmo-operator-backend-worker"
  osmo_workflows_namespace = "osmo-workflows"
  osmo_workflow_sa         = "osmo-workflow"
  aws_lb_controller_sa     = "aws-load-balancer-controller"
  cluster_autoscaler_sa    = "cluster-autoscaler"
  external_secrets_sa      = "external-secrets"

  # GPU node group taints
  # EKS automatically applies eks.amazonaws.com/capacityType=SPOT taint
  # to Spot nodes, so only the nvidia.com/gpu taint needs to be set manually.
  gpu_taints = [
    {
      key    = "nvidia.com/gpu"
      value  = "true"
      effect = "NO_SCHEDULE"
    }
  ]
}
