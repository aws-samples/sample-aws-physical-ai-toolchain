# SPDX-License-Identifier: Apache-2.0

#------------------------------------------------------------------------------
# Platform Module - Shared AWS Services
#------------------------------------------------------------------------------

module "platform" {
  source = "./modules/platform"

  providers = {
    aws           = aws
    aws.us_east_1 = aws.us_east_1
  }

  # Core configuration
  name_prefix = local.name_prefix
  environment = var.environment
  aws_region  = var.aws_region
  common_tags = local.common_tags

  # DNS / ACM configuration
  route53_zone_id    = var.route53_zone_id
  osmo_hostname      = local.osmo_hostname
  osmo_auth_hostname = local.osmo_auth_hostname

  # VPC configuration
  vpc_cidr            = var.vpc_cidr
  azs                 = local.azs
  private_subnets     = local.private_subnets
  public_subnets      = local.public_subnets
  database_subnets    = local.database_subnets
  elasticache_subnets = local.elasticache_subnets
  single_nat_gateway  = var.single_nat_gateway
  cluster_name        = var.cluster_name

  # RDS PostgreSQL configuration
  deploy_postgresql           = local.deploy_postgresql
  rds_instance_class          = var.rds_instance_class
  rds_engine_version          = var.rds_engine_version
  rds_allocated_storage       = var.rds_allocated_storage
  rds_max_allocated_storage   = var.rds_max_allocated_storage
  rds_db_name                 = var.rds_db_name
  rds_username                = var.rds_username
  rds_multi_az                = var.rds_multi_az
  rds_backup_retention_period = var.rds_backup_retention_period
  rds_deletion_protection     = var.rds_deletion_protection

  # ElastiCache Redis configuration
  deploy_redis                     = local.deploy_redis
  redis_node_type                  = var.redis_node_type
  redis_engine_version             = var.redis_engine_version
  redis_num_cache_clusters         = var.redis_num_cache_clusters
  redis_automatic_failover_enabled = var.redis_automatic_failover_enabled
  redis_multi_az_enabled           = var.redis_multi_az_enabled
  redis_snapshot_retention_limit   = var.redis_snapshot_retention_limit

  # S3 configuration
  s3_workflows_bucket   = local.s3_workflows_bucket
  s3_datasets_bucket    = local.s3_datasets_bucket
  s3_force_destroy      = var.s3_force_destroy
  s3_versioning_enabled = var.s3_versioning_enabled

  # KMS configuration
  kms_key_deletion_window_days = var.kms_key_deletion_window_days
  kms_enable_key_rotation      = var.kms_enable_key_rotation

  # Secrets Manager configuration
  secrets_recovery_window_days = var.secrets_recovery_window_days

  # WAF configuration
  enable_waf           = var.enable_waf
  waf_allowed_ip_cidrs = var.waf_allowed_ip_cidrs
  waf_rate_limit       = var.waf_rate_limit
  waf_block_mode       = var.waf_block_mode

  # Enhanced security configuration
  enable_flow_logs           = var.enable_flow_logs
  flow_logs_retention_days   = var.flow_logs_retention_days
  enable_guardduty           = var.enable_guardduty
  enable_security_hub        = var.enable_security_hub
  restrict_rds_to_eks_only   = var.restrict_rds_to_eks_only
  restrict_redis_to_eks_only = var.restrict_redis_to_eks_only
  s3_allowed_origins         = var.s3_allowed_origins

  # ALB access restriction
  alb_allowed_cidrs = var.alb_allowed_cidrs

  # Cognito configuration
  deploy_cognito                       = var.deploy_cognito
  cognito_parent_domain_placeholder_ip = var.cognito_parent_domain_placeholder_ip
  cognito_admin_username               = var.cognito_admin_username
  cognito_admin_email                  = var.cognito_admin_email
  cognito_admin_temp_password          = var.cognito_admin_temp_password

  # Identity Center configuration (legacy)
  deploy_identity_center = var.deploy_identity_center
  idc_region             = var.idc_region
  idc_client_secret      = var.idc_client_secret

  # Keycloak configuration
  deploy_keycloak = var.deploy_keycloak
  keycloak_realm  = var.keycloak_realm
}

#------------------------------------------------------------------------------
# EKS Module - Kubernetes Cluster
# EKS is created only after the entire platform module completes (see depends_on
# and module.platform outputs). So RDS, Redis, Cognito, etc. and the cluster do
# not run in parallel. To run the cluster in parallel with RDS/Redis/Cognito,
# split platform into a "foundation" module (VPC, KMS, S3, Secrets ARNs) and have
# EKS depend only on foundation; the rest of platform can then run alongside EKS.
#------------------------------------------------------------------------------

module "eks" {
  source = "./modules/eks"

  # Core configuration
  name_prefix  = local.name_prefix
  common_tags  = local.common_tags
  cluster_name = var.cluster_name

  # VPC configuration
  vpc_id          = module.platform.vpc_id
  private_subnets = module.platform.private_subnet_ids

  # EKS cluster configuration
  kubernetes_version              = var.kubernetes_version
  cluster_endpoint_public_access  = var.cluster_endpoint_public_access
  cluster_endpoint_private_access = var.cluster_endpoint_private_access
  eks_admin_principal_arns        = var.eks_admin_principal_arns

  # System node group configuration
  system_node_instance_types = var.system_node_instance_types
  system_node_min_size       = var.system_node_min_size
  system_node_max_size       = var.system_node_max_size
  system_node_desired_size   = var.system_node_desired_size

  # GPU node group configuration
  deploy_gpu_nodes        = local.deploy_gpu_nodes
  gpu_node_instance_types = var.gpu_node_instance_types
  gpu_node_min_size       = var.gpu_node_min_size
  gpu_node_max_size       = var.gpu_node_max_size
  gpu_node_desired_size   = var.gpu_node_desired_size
  gpu_node_capacity_type  = var.gpu_node_capacity_type
  gpu_ami_type            = var.gpu_ami_type
  gpu_ami_id              = var.gpu_ami_id
  gpu_taints              = local.gpu_taints
  gpu_node_volume_size    = var.gpu_node_volume_size

  # IRSA configuration
  osmo_namespace           = local.osmo_namespace
  osmo_service_account     = local.osmo_service_account
  osmo_operator_namespace  = local.osmo_operator_namespace
  osmo_backend_listener_sa = local.osmo_backend_listener_sa
  osmo_backend_worker_sa   = local.osmo_backend_worker_sa
  osmo_workflows_namespace = local.osmo_workflows_namespace
  osmo_workflow_sa         = local.osmo_workflow_sa
  aws_lb_controller_sa     = local.aws_lb_controller_sa
  cluster_autoscaler_sa    = local.cluster_autoscaler_sa
  external_secrets_sa      = local.external_secrets_sa

  # Route53 for external-dns IRSA
  route53_zone_id = var.route53_zone_id

  # S3 bucket ARNs for IRSA policies
  s3_workflows_bucket_arn = module.platform.s3_workflows_bucket_arn
  s3_datasets_bucket_arn  = module.platform.s3_datasets_bucket_arn

  # Secrets Manager ARN for IRSA policies
  secrets_manager_arn = module.platform.secrets_manager_arn

  # KMS key ARN for IRSA policies
  kms_key_arn = module.platform.kms_key_arn

  depends_on = [module.platform]
}

#------------------------------------------------------------------------------
# Cross-Module Security Group Rules
# Defined at root to avoid circular dependency between platform and EKS modules
#------------------------------------------------------------------------------

resource "aws_security_group_rule" "rds_from_eks" {
  count = local.deploy_postgresql ? 1 : 0

  type                     = "ingress"
  from_port                = 5432
  to_port                  = 5432
  protocol                 = "tcp"
  source_security_group_id = module.eks.cluster_primary_security_group_id
  security_group_id        = module.platform.rds_security_group_id
  description              = "PostgreSQL from EKS cluster"
}

resource "aws_security_group_rule" "redis_from_eks" {
  count = local.deploy_redis ? 1 : 0

  type                     = "ingress"
  from_port                = 6379
  to_port                  = 6379
  protocol                 = "tcp"
  source_security_group_id = module.eks.cluster_primary_security_group_id
  security_group_id        = module.platform.redis_security_group_id
  description              = "Redis from EKS cluster"
}

resource "aws_security_group_rule" "nodes_from_alb" {
  count = length(var.alb_allowed_cidrs) > 0 ? 1 : 0

  type                     = "ingress"
  from_port                = 1
  to_port                  = 65535
  protocol                 = "tcp"
  source_security_group_id = module.platform.alb_security_group_id
  security_group_id        = module.eks.node_security_group_id
  description              = "ALB health checks and traffic to pod targets"
}
