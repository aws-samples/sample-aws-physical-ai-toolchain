# SPDX-License-Identifier: Apache-2.0

#------------------------------------------------------------------------------
# Core Configuration
#------------------------------------------------------------------------------

variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "cluster_name" {
  description = "Name for the EKS cluster and related resources"
  type        = string

  validation {
    condition     = length(var.cluster_name) >= 1 && length(var.cluster_name) <= 20
    error_message = "cluster_name must be 1-20 characters."
  }
}

variable "resource_suffix" {
  description = "Suffix for resource names to avoid collisions (e.g. osm01, dev01). Use a unique value per deployment when sharing the same cluster_name and environment. Required - no default."
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9]{1,8}$", var.resource_suffix))
    error_message = "resource_suffix must be 1-8 lowercase alphanumeric characters (e.g. osm01)."
  }
}

variable "project_name" {
  description = "Project name for resource tagging"
  type        = string
  default     = "osmo"
}

variable "owner" {
  description = "Owner tag for resources"
  type        = string
  default     = "osmo-team"
}

#------------------------------------------------------------------------------
# Deployment Mode
#------------------------------------------------------------------------------

variable "deployment_mode" {
  description = "Deployment mode: full, control-plane-only, or backend-only"
  type        = string
  default     = "full"

  validation {
    condition     = contains(["full", "control-plane-only", "backend-only"], var.deployment_mode)
    error_message = "Deployment mode must be one of: full, control-plane-only, backend-only."
  }
}

variable "external_service_url" {
  description = "External OSMO service URL (required for backend-only mode)"
  type        = string
  default     = null
}

variable "route53_zone_id" {
  description = "ID of existing Route53 hosted zone (e.g., ZEXAMPLEZONEID123456)"
  type        = string
}

variable "route53_zone_name" {
  description = "Name of existing Route53 hosted zone (e.g., example.com)"
  type        = string
}

#------------------------------------------------------------------------------
# VPC Configuration
#------------------------------------------------------------------------------

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones_count" {
  description = "Number of availability zones to use"
  type        = number
  default     = 3

  validation {
    condition     = var.availability_zones_count >= 2 && var.availability_zones_count <= 3
    error_message = "Availability zones count must be between 2 and 3."
  }
}

variable "single_nat_gateway" {
  description = "Use a single NAT gateway (cost savings for non-prod)"
  type        = bool
  default     = true
}

#------------------------------------------------------------------------------
# EKS Configuration
#------------------------------------------------------------------------------

variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.35"
}

variable "cluster_endpoint_public_access" {
  description = "Enable public access to EKS API endpoint"
  type        = bool
  default     = true
}

variable "cluster_endpoint_private_access" {
  description = "Enable private access to EKS API endpoint"
  type        = bool
  default     = true
}

variable "eks_admin_principal_arns" {
  description = "List of IAM principal ARNs to grant EKS admin access"
  type        = list(string)
  default     = []
}

#------------------------------------------------------------------------------
# System Node Group Configuration
#------------------------------------------------------------------------------

variable "system_node_instance_types" {
  description = "Instance types for system node group"
  type        = list(string)
  default     = ["m6i.xlarge"]
}

variable "system_node_min_size" {
  description = "Minimum number of system nodes"
  type        = number
  default     = 2
}

variable "system_node_max_size" {
  description = "Maximum number of system nodes"
  type        = number
  default     = 6
}

variable "system_node_desired_size" {
  description = "Desired number of system nodes"
  type        = number
  default     = 3
}

#------------------------------------------------------------------------------
# GPU Node Group Configuration
#------------------------------------------------------------------------------

variable "should_deploy_gpu_nodes" {
  description = "Deploy GPU node group"
  type        = bool
  default     = true
}

variable "gpu_node_instance_types" {
  description = "Instance types for GPU node group"
  type        = list(string)
  default     = ["g5.2xlarge"]
}

variable "gpu_node_min_size" {
  description = "Minimum number of GPU nodes"
  type        = number
  default     = 0
}

variable "gpu_node_max_size" {
  description = "Maximum number of GPU nodes"
  type        = number
  default     = 10
}

variable "gpu_node_desired_size" {
  description = "Desired number of GPU nodes"
  type        = number
  default     = 0
}

variable "gpu_node_capacity_type" {
  description = "Capacity type for GPU nodes (ON_DEMAND or SPOT)"
  type        = string
  default     = "ON_DEMAND"

  validation {
    condition     = contains(["ON_DEMAND", "SPOT"], var.gpu_node_capacity_type)
    error_message = "Capacity type must be ON_DEMAND or SPOT."
  }
}

variable "gpu_ami_type" {
  description = "AMI type for GPU nodes (ignored when gpu_ami_id is set)"
  type        = string
  default     = "AL2_x86_64_GPU"
}

variable "gpu_ami_id" {
  description = "Required. AMI ID for GPU nodes — the Canonical Ubuntu EKS image for your region and Kubernetes version (see https://cloud-images.ubuntu.com/docs/aws/eks/). gpu_ami_type is ignored and the NVIDIA GPU Operator handles driver installation."
  type        = string
}

variable "gpu_node_volume_size" {
  description = "Root EBS volume size (GiB) for GPU nodes. The 200 GiB default upstream is too small to hold the Isaac Sim (~30 GB) + Cosmos Predict2 (~25 GB) container images alongside the GPU operator driver containers — pods get evicted with 'no space left on device' during image extraction. Must be at least 200 GiB."
  type        = number
  default     = 500
}

#------------------------------------------------------------------------------
# RDS PostgreSQL Configuration
#------------------------------------------------------------------------------

variable "should_deploy_postgresql" {
  description = "Deploy RDS PostgreSQL instance"
  type        = bool
  default     = true
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "rds_engine_version" {
  description = "PostgreSQL engine version"
  type        = string
  default     = "15.17"
}

variable "rds_allocated_storage" {
  description = "Allocated storage in GB"
  type        = number
  default     = 20
}

variable "rds_max_allocated_storage" {
  description = "Maximum allocated storage in GB (autoscaling)"
  type        = number
  default     = 100
}

variable "rds_db_name" {
  description = "Name of the database to create"
  type        = string
  default     = "osmo"
}

variable "rds_username" {
  description = "Master username for RDS"
  type        = string
  default     = "osmo_admin"
}

variable "rds_multi_az" {
  description = "Enable Multi-AZ deployment"
  type        = bool
  default     = false
}

variable "rds_backup_retention_period" {
  description = "Backup retention period in days"
  type        = number
  default     = 7
}

variable "rds_deletion_protection" {
  description = "Enable deletion protection"
  type        = bool
  default     = false
}

#------------------------------------------------------------------------------
# ElastiCache Redis Configuration
#------------------------------------------------------------------------------

variable "should_deploy_redis" {
  description = "Deploy ElastiCache Redis cluster"
  type        = bool
  default     = true
}

variable "redis_node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.t3.medium"
}

variable "redis_engine_version" {
  description = "Redis engine version"
  type        = string
  default     = "7.1"
}

variable "redis_num_cache_clusters" {
  description = "Number of cache clusters (nodes) in the replication group"
  type        = number
  default     = 1
}

variable "redis_automatic_failover_enabled" {
  description = "Enable automatic failover (requires num_cache_clusters >= 2)"
  type        = bool
  default     = false
}

variable "redis_multi_az_enabled" {
  description = "Enable Multi-AZ for Redis"
  type        = bool
  default     = false
}

variable "redis_snapshot_retention_limit" {
  description = "Number of days to retain automatic snapshots"
  type        = number
  default     = 1
}

#------------------------------------------------------------------------------
# S3 Configuration
#------------------------------------------------------------------------------

variable "s3_force_destroy" {
  description = "Allow S3 buckets to be destroyed with objects inside"
  type        = bool
  default     = false
}

variable "s3_versioning_enabled" {
  description = "Enable versioning for S3 buckets"
  type        = bool
  default     = true
}

#------------------------------------------------------------------------------
# KMS Configuration
#------------------------------------------------------------------------------

variable "kms_key_deletion_window_days" {
  description = "Waiting period before KMS key deletion"
  type        = number
  default     = 7
}

variable "kms_enable_key_rotation" {
  description = "Enable automatic key rotation for KMS keys"
  type        = bool
  default     = true
}

#------------------------------------------------------------------------------
# Secrets Manager Configuration
#------------------------------------------------------------------------------

variable "secrets_recovery_window_days" {
  description = "Number of days AWS Secrets Manager waits before deletion"
  type        = number
  default     = 7
}

#------------------------------------------------------------------------------
# WAF Configuration (Production Security)
#------------------------------------------------------------------------------

variable "enable_waf" {
  description = "Enable AWS WAF for ALB protection"
  type        = bool
  default     = false
}

variable "waf_allowed_ip_cidrs" {
  description = "List of CIDR blocks allowed to access the ALB (empty = allow all)"
  type        = list(string)
  default     = []
}

variable "waf_rate_limit" {
  description = "Rate limit for requests per 5-minute period per IP"
  type        = number
  default     = 2000
}

variable "waf_block_mode" {
  description = "WAF action mode: COUNT (monitor) or BLOCK (enforce)"
  type        = string
  default     = "BLOCK"

  validation {
    condition     = contains(["COUNT", "BLOCK"], var.waf_block_mode)
    error_message = "WAF block mode must be COUNT or BLOCK."
  }
}

#------------------------------------------------------------------------------
# Enhanced Security Configuration
#------------------------------------------------------------------------------

variable "enable_flow_logs" {
  description = "Enable VPC Flow Logs for network traffic analysis"
  type        = bool
  default     = false
}

variable "flow_logs_retention_days" {
  description = "Retention period for VPC Flow Logs in CloudWatch"
  type        = number
  default     = 30
}

variable "enable_guardduty" {
  description = "Enable AWS GuardDuty for threat detection"
  type        = bool
  default     = false
}

variable "enable_security_hub" {
  description = "Enable AWS Security Hub for security posture management"
  type        = bool
  default     = false
}

#------------------------------------------------------------------------------
# Observability Configuration
#------------------------------------------------------------------------------

variable "enable_cloudwatch_logging" {
  description = "Provision the Fluent Bit IRSA role + CloudWatch log group for shipping pod logs. The Fluent Bit DaemonSet itself is deployed by 01-deploy-aws-prerequisites.sh."
  type        = bool
  default     = true
}

variable "cloudwatch_logs_retention_days" {
  description = "Retention period (days) for the OSMO CloudWatch log group"
  type        = number
  default     = 30
}

variable "enable_managed_prometheus" {
  description = "Provision an Amazon Managed Prometheus (AMP) workspace + managed scraper for cluster/GPU metrics"
  type        = bool
  default     = true
}

variable "restrict_rds_to_eks_only" {
  description = "Restrict RDS access to EKS security group only (no VPC CIDR access)"
  type        = bool
  default     = false
}

variable "restrict_redis_to_eks_only" {
  description = "Restrict Redis access to EKS security group only (no VPC CIDR access)"
  type        = bool
  default     = false
}

variable "s3_allowed_origins" {
  description = "Allowed origins for S3 CORS (empty = allow all, production should specify domains)"
  type        = list(string)
  default     = []
}

#------------------------------------------------------------------------------
# ALB Access Restriction
#------------------------------------------------------------------------------

variable "alb_allowed_cidrs" {
  description = "CIDRs allowed to access the ALB on port 443 (e.g. your IP: [\"1.2.3.4/32\"]). Empty = no SG created, ALB controller manages its own."
  type        = list(string)
  default     = []
}

#------------------------------------------------------------------------------
# Cognito Configuration (legacy — set deploy_cognito = false when using Identity Center)
#------------------------------------------------------------------------------

variable "deploy_cognito" {
  description = "Deploy AWS Cognito User Pool as the OSMO identity provider"
  type        = bool
  default     = false
}

variable "cognito_parent_domain_placeholder_ip" {
  description = "If set, Terraform creates an A record for the parent domain of the Cognito auth hostname so Cognito custom domain validation passes (e.g. 192.0.2.1). Leave empty if the parent domain already has an A record."
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
  description = "Temporary password for the initial Cognito admin user (must change on first login)"
  type        = string
  sensitive   = true
  default     = "ChangeMe123!"
}

#------------------------------------------------------------------------------
# AWS IAM Identity Center Configuration (legacy — set deploy_identity_center = false when using Keycloak)
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
  description = "OAuth2 client secret — generated in the Identity Center console after terraform apply. Leave empty on first apply."
  type        = string
  sensitive   = true
  default     = ""
}

variable "idc_admin_email" {
  description = "Email of the admin user in Identity Center (legacy)"
  type        = string
  default     = ""
}

#------------------------------------------------------------------------------
# Keycloak Configuration
#------------------------------------------------------------------------------

variable "deploy_keycloak" {
  description = "Provision ACM certificate and Terraform outputs for a self-hosted Keycloak IdP"
  type        = bool
  default     = true
}

variable "keycloak_realm" {
  description = "Keycloak realm name for OSMO"
  type        = string
  default     = "osmo"
}

variable "keycloak_admin_username" {
  description = "Username of the OSMO admin user inside Keycloak (used as x-osmo-user identity)"
  type        = string
  default     = "admin"
}
