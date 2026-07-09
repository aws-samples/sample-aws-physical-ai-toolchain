# SPDX-License-Identifier: Apache-2.0

#------------------------------------------------------------------------------
# Core Configuration
#------------------------------------------------------------------------------

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "common_tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
}

variable "route53_zone_id" {
  description = "ID of existing Route53 hosted zone"
  type        = string
}

variable "osmo_hostname" {
  description = "FQDN for OSMO service (e.g., osmo-aws.example.com)"
  type        = string
}

variable "osmo_auth_hostname" {
  description = "FQDN for OSMO auth/Keycloak (e.g., osmo-aws-auth.example.com)"
  type        = string
}

variable "cluster_name" {
  description = "EKS cluster name for tagging"
  type        = string
}

#------------------------------------------------------------------------------
# VPC Configuration
#------------------------------------------------------------------------------

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
}

variable "azs" {
  description = "List of availability zones"
  type        = list(string)
}

variable "private_subnets" {
  description = "Private subnet CIDRs"
  type        = list(string)
}

variable "public_subnets" {
  description = "Public subnet CIDRs"
  type        = list(string)
}

variable "database_subnets" {
  description = "Database subnet CIDRs"
  type        = list(string)
}

variable "elasticache_subnets" {
  description = "ElastiCache subnet CIDRs"
  type        = list(string)
}

variable "single_nat_gateway" {
  description = "Use single NAT gateway"
  type        = bool
}

#------------------------------------------------------------------------------
# RDS Configuration
#------------------------------------------------------------------------------

variable "deploy_postgresql" {
  description = "Deploy RDS PostgreSQL"
  type        = bool
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
}

variable "rds_engine_version" {
  description = "PostgreSQL engine version"
  type        = string
}

variable "rds_allocated_storage" {
  description = "Allocated storage in GB"
  type        = number
}

variable "rds_max_allocated_storage" {
  description = "Maximum allocated storage in GB"
  type        = number
}

variable "rds_db_name" {
  description = "Database name"
  type        = string
}

variable "rds_username" {
  description = "Master username"
  type        = string
}

variable "rds_multi_az" {
  description = "Enable Multi-AZ"
  type        = bool
}

variable "rds_backup_retention_period" {
  description = "Backup retention period"
  type        = number
}

variable "rds_deletion_protection" {
  description = "Enable deletion protection"
  type        = bool
}

#------------------------------------------------------------------------------
# ElastiCache Redis Configuration
#------------------------------------------------------------------------------

variable "deploy_redis" {
  description = "Deploy ElastiCache Redis"
  type        = bool
}

variable "redis_node_type" {
  description = "ElastiCache node type"
  type        = string
}

variable "redis_engine_version" {
  description = "Redis engine version"
  type        = string
}

variable "redis_num_cache_clusters" {
  description = "Number of cache clusters"
  type        = number
}

variable "redis_automatic_failover_enabled" {
  description = "Enable automatic failover"
  type        = bool
}

variable "redis_multi_az_enabled" {
  description = "Enable Multi-AZ"
  type        = bool
}

variable "redis_snapshot_retention_limit" {
  description = "Snapshot retention limit"
  type        = number
}

#------------------------------------------------------------------------------
# S3 Configuration
#------------------------------------------------------------------------------

variable "s3_workflows_bucket" {
  description = "S3 bucket name for workflows"
  type        = string
}

variable "s3_datasets_bucket" {
  description = "S3 bucket name for datasets"
  type        = string
}

variable "s3_force_destroy" {
  description = "Allow bucket deletion with objects"
  type        = bool
}

variable "s3_versioning_enabled" {
  description = "Enable bucket versioning"
  type        = bool
}

#------------------------------------------------------------------------------
# KMS Configuration
#------------------------------------------------------------------------------

variable "kms_key_deletion_window_days" {
  description = "KMS key deletion window"
  type        = number
}

variable "kms_enable_key_rotation" {
  description = "Enable KMS key rotation"
  type        = bool
}

#------------------------------------------------------------------------------
# Secrets Manager Configuration
#------------------------------------------------------------------------------

variable "secrets_recovery_window_days" {
  description = "Secrets Manager recovery window"
  type        = number
}

#------------------------------------------------------------------------------
# WAF Configuration
#------------------------------------------------------------------------------

variable "enable_waf" {
  description = "Enable AWS WAF for ALB protection"
  type        = bool
  default     = false
}

variable "waf_allowed_ip_cidrs" {
  description = "List of CIDR blocks allowed to access the ALB"
  type        = list(string)
  default     = []
}

variable "waf_rate_limit" {
  description = "Rate limit for requests per 5-minute period per IP"
  type        = number
  default     = 2000
}

variable "waf_block_mode" {
  description = "WAF action mode: COUNT or BLOCK"
  type        = string
  default     = "BLOCK"
}

#------------------------------------------------------------------------------
# Enhanced Security Configuration
#------------------------------------------------------------------------------

variable "enable_flow_logs" {
  description = "Enable VPC Flow Logs"
  type        = bool
  default     = false
}

variable "flow_logs_retention_days" {
  description = "Retention period for VPC Flow Logs"
  type        = number
  default     = 30
}

variable "enable_guardduty" {
  description = "Enable AWS GuardDuty"
  type        = bool
  default     = false
}

variable "enable_security_hub" {
  description = "Enable AWS Security Hub"
  type        = bool
  default     = false
}

variable "restrict_rds_to_eks_only" {
  description = "Restrict RDS access to EKS security group only"
  type        = bool
  default     = false
}

variable "restrict_redis_to_eks_only" {
  description = "Restrict Redis access to EKS security group only"
  type        = bool
  default     = false
}

variable "s3_allowed_origins" {
  description = "Allowed origins for S3 CORS"
  type        = list(string)
  default     = []
}

#------------------------------------------------------------------------------
# ALB Access Restriction
#------------------------------------------------------------------------------

variable "alb_allowed_cidrs" {
  description = "CIDRs allowed to access the ALB on port 443. Empty = no SG created."
  type        = list(string)
  default     = []
}

#------------------------------------------------------------------------------
# Cognito Configuration
#------------------------------------------------------------------------------

variable "deploy_cognito" {
  description = "Deploy AWS Cognito User Pool as the OSMO identity provider"
  type        = bool
  default     = true
}

# Cognito requires the parent domain of the custom domain to have a resolvable A (or AAAA)
# record. E.g. for osmo-aws-auth.example.com the parent is example.com. Set this
# to a placeholder IP (e.g. 192.0.2.1) to have Terraform create an A record for the parent
# in the same Route53 zone so Cognito's check passes; leave empty if the parent already has
# an A record or you manage it elsewhere.
variable "cognito_parent_domain_placeholder_ip" {
  description = "If set, create an A record for the parent domain of osmo_auth_hostname so Cognito custom domain validation passes (e.g. 192.0.2.1). Leave empty if parent domain already has an A record."
  type        = string
  default     = "192.0.2.1"
}

variable "cognito_admin_username" {
  description = "Username for the initial Cognito admin user"
  type        = string
  default     = "osmo-admin"
}

variable "cognito_admin_email" {
  description = "Email for the initial Cognito admin user"
  type        = string
  default     = ""
}

variable "cognito_admin_temp_password" {
  description = "Temporary password for the initial Cognito admin user (user must change on first login)"
  type        = string
  sensitive   = true
  default     = "ChangeMe123!"
}

#------------------------------------------------------------------------------
# AWS IAM Identity Center Configuration (legacy)
#------------------------------------------------------------------------------

variable "deploy_identity_center" {
  description = "Create an IAM Identity Center OAuth 2.0 application for OSMO"
  type        = bool
  default     = false
}

variable "idc_region" {
  description = "AWS region where Identity Center is enabled (defaults to aws_region)"
  type        = string
  default     = ""
}

variable "idc_client_secret" {
  description = "OAuth2 client secret — generated in the Identity Center console after terraform apply"
  type        = string
  sensitive   = true
  default     = ""
}

#------------------------------------------------------------------------------
# Keycloak Configuration
#------------------------------------------------------------------------------

variable "deploy_keycloak" {
  description = "Provision ACM certificate and outputs for self-hosted Keycloak"
  type        = bool
  default     = true
}

variable "keycloak_realm" {
  description = "Keycloak realm name for OSMO"
  type        = string
  default     = "osmo"
}
