# SPDX-License-Identifier: Apache-2.0

#------------------------------------------------------------------------------
# Cluster Information
#------------------------------------------------------------------------------

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "EKS cluster API endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_ca_certificate" {
  description = "Base64 encoded cluster CA certificate"
  value       = module.eks.cluster_ca_certificate
  sensitive   = true
}

output "cluster_oidc_issuer_url" {
  description = "OIDC issuer URL for the EKS cluster"
  value       = module.eks.cluster_oidc_issuer_url
}

output "cluster_oidc_provider_arn" {
  description = "OIDC provider ARN for IRSA"
  value       = module.eks.oidc_provider_arn
}

#------------------------------------------------------------------------------
# VPC Information
#------------------------------------------------------------------------------

output "vpc_id" {
  description = "VPC ID"
  value       = module.platform.vpc_id
}

output "private_subnet_ids" {
  description = "List of private subnet IDs"
  value       = module.platform.private_subnet_ids
}

output "public_subnet_ids" {
  description = "List of public subnet IDs"
  value       = module.platform.public_subnet_ids
}

#------------------------------------------------------------------------------
# Database Outputs
#------------------------------------------------------------------------------

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = module.platform.rds_endpoint
}

output "rds_port" {
  description = "RDS instance port"
  value       = module.platform.rds_port
}

output "rds_database_name" {
  description = "RDS database name"
  value       = module.platform.rds_database_name
}

output "rds_username" {
  description = "RDS master username"
  value       = module.platform.rds_username
  sensitive   = true
}

output "rds_password_secret_arn" {
  description = "ARN of the Secrets Manager secret containing RDS password"
  value       = module.platform.rds_password_secret_arn
}

#------------------------------------------------------------------------------
# Redis Outputs
#------------------------------------------------------------------------------

output "redis_endpoint" {
  description = "Redis primary endpoint"
  value       = module.platform.redis_endpoint
}

output "redis_port" {
  description = "Redis port"
  value       = module.platform.redis_port
}

output "redis_auth_token_secret_arn" {
  description = "ARN of the Secrets Manager secret containing Redis auth token"
  value       = module.platform.redis_auth_token_secret_arn
}

#------------------------------------------------------------------------------
# S3 Outputs
#------------------------------------------------------------------------------

output "s3_workflows_bucket_name" {
  description = "S3 bucket name for workflows"
  value       = module.platform.s3_workflows_bucket_name
}

output "s3_workflows_bucket_arn" {
  description = "S3 bucket ARN for workflows"
  value       = module.platform.s3_workflows_bucket_arn
}

output "s3_datasets_bucket_name" {
  description = "S3 bucket name for datasets"
  value       = module.platform.s3_datasets_bucket_name
}

output "s3_datasets_bucket_arn" {
  description = "S3 bucket ARN for datasets"
  value       = module.platform.s3_datasets_bucket_arn
}

output "s3_region" {
  description = "S3 region"
  value       = var.aws_region
}

#------------------------------------------------------------------------------
# KMS Outputs
#------------------------------------------------------------------------------

output "kms_key_arn" {
  description = "KMS key ARN for encryption"
  value       = module.platform.kms_key_arn
}

output "kms_key_id" {
  description = "KMS key ID"
  value       = module.platform.kms_key_id
}

#------------------------------------------------------------------------------
# Secrets Manager Outputs
#------------------------------------------------------------------------------

output "secrets_manager_arn" {
  description = "Secrets Manager secret ARN for OSMO secrets"
  value       = module.platform.secrets_manager_arn
}

output "osmo_s3_credentials_secret_arn" {
  description = "ARN of the Secrets Manager secret containing OSMO S3 IAM user credentials"
  value       = module.platform.osmo_s3_credentials_secret_arn
}

#------------------------------------------------------------------------------
# IRSA Role ARNs
#------------------------------------------------------------------------------

output "osmo_service_role_arn" {
  description = "IAM role ARN for OSMO service"
  value       = module.eks.osmo_service_role_arn
}

output "osmo_backend_role_arn" {
  description = "IAM role ARN for OSMO backend operator"
  value       = module.eks.osmo_backend_role_arn
}

output "osmo_workflow_role_arn" {
  description = "IAM role ARN for OSMO workflow task pods (IRSA dataset S3 access)"
  value       = module.eks.osmo_workflow_role_arn
}

#------------------------------------------------------------------------------
# Observability Outputs
#------------------------------------------------------------------------------

output "fluentbit_role_arn" {
  description = "IAM role ARN for the Fluent Bit log shipper (IRSA)"
  value       = var.enable_cloudwatch_logging ? module.fluentbit_irsa[0].iam_role_arn : null
}

output "cloudwatch_log_group_name" {
  description = "CloudWatch log group for OSMO container logs"
  value       = var.enable_cloudwatch_logging ? aws_cloudwatch_log_group.osmo_logs[0].name : null
}

output "amp_workspace_id" {
  description = "Amazon Managed Prometheus workspace ID"
  value       = var.enable_managed_prometheus ? aws_prometheus_workspace.osmo[0].id : null
}

output "amp_workspace_prometheus_endpoint" {
  description = "Amazon Managed Prometheus remote-write/query endpoint"
  value       = var.enable_managed_prometheus ? aws_prometheus_workspace.osmo[0].prometheus_endpoint : null
}

output "aws_lb_controller_role_arn" {
  description = "IAM role ARN for AWS Load Balancer Controller"
  value       = module.eks.aws_lb_controller_role_arn
}

output "ebs_csi_driver_role_arn" {
  description = "IAM role ARN for EBS CSI driver"
  value       = module.eks.ebs_csi_driver_role_arn
}

output "cluster_autoscaler_role_arn" {
  description = "IAM role ARN for Cluster Autoscaler"
  value       = module.eks.cluster_autoscaler_role_arn
}

output "external_secrets_role_arn" {
  description = "IAM role ARN for External Secrets Operator"
  value       = module.eks.external_secrets_role_arn
}

#------------------------------------------------------------------------------
# Configuration Outputs for Scripts
#------------------------------------------------------------------------------

output "aws_region" {
  description = "AWS region"
  value       = var.aws_region
}

output "environment" {
  description = "Environment name"
  value       = var.environment
}

output "deployment_mode" {
  description = "Deployment mode"
  value       = var.deployment_mode
}

output "resource_suffix" {
  description = "Resource suffix used for naming (avoids collisions between deployments)"
  value       = var.resource_suffix
}

output "external_service_url" {
  description = "External OSMO service URL (for backend-only mode)"
  value       = var.external_service_url
}

output "osmo_hostname" {
  description = "Hostname for OSMO service Ingress"
  value       = local.osmo_hostname
}

output "osmo_auth_hostname" {
  description = "Hostname for OSMO auth/Keycloak Ingress"
  value       = local.osmo_auth_hostname
}

output "acm_certificate_arn" {
  description = "ACM certificate ARN for OSMO service domain"
  value       = module.platform.acm_certificate_arn
}

output "acm_auth_certificate_arn" {
  description = "ACM certificate ARN for OSMO auth domain"
  value       = module.platform.acm_auth_certificate_arn
}

output "external_dns_role_arn" {
  description = "IAM role ARN for external-dns"
  value       = module.eks.external_dns_role_arn
}

output "route53_zone_id" {
  description = "Route53 hosted zone ID"
  value       = var.route53_zone_id
}

output "email_security_protected_domains" {
  description = "Domains protected from email spoofing (DMARC p=reject on each; SPF v=spf1 -all on the apex). Validate: dig +short TXT <domain> ; dig +short TXT _dmarc.<domain>"
  value       = module.platform.email_security_protected_domains
}

output "osmo_namespace" {
  description = "Kubernetes namespace for OSMO"
  value       = local.osmo_namespace
}

output "osmo_service_account" {
  description = "Service account name for OSMO service"
  value       = local.osmo_service_account
}

output "osmo_backend_service_account" {
  description = "Service account name for OSMO backend operator"
  value       = local.osmo_backend_sa
}

output "osmo_workflows_namespace" {
  description = "Kubernetes namespace where OSMO workflow task pods run"
  value       = local.osmo_workflows_namespace
}

output "osmo_workflow_service_account" {
  description = "Service account name used by OSMO workflow task pods (IRSA)"
  value       = local.osmo_workflow_sa
}

#------------------------------------------------------------------------------
# WAF Outputs
#------------------------------------------------------------------------------

output "waf_web_acl_arn" {
  description = "WAF Web ACL ARN (for ALB association)"
  value       = module.platform.waf_web_acl_arn
}

output "alb_security_group_id" {
  description = "ALB security group ID (null when alb_allowed_cidrs is empty)"
  value       = module.platform.alb_security_group_id
}

#------------------------------------------------------------------------------
# kubectl Configuration Command
#------------------------------------------------------------------------------

output "configure_kubectl" {
  description = "AWS CLI command to configure kubectl"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}

#------------------------------------------------------------------------------
# Cognito Outputs (legacy — null when deploy_cognito = false)
#------------------------------------------------------------------------------

output "cognito_user_pool_id" {
  description = "Cognito User Pool ID"
  value       = module.platform.cognito_user_pool_id
}

output "cognito_issuer_url" {
  description = "Cognito OIDC issuer URL"
  value       = module.platform.cognito_issuer_url
}

output "cognito_jwks_uri" {
  description = "Cognito JWKS URI for JWT validation"
  value       = module.platform.cognito_jwks_uri
}

output "cognito_browser_client_id" {
  description = "Cognito browser app client ID"
  value       = module.platform.cognito_browser_client_id
}

output "cognito_browser_client_secret" {
  description = "Cognito browser app client secret"
  value       = module.platform.cognito_browser_client_secret
  sensitive   = true
}

output "cognito_cli_client_id" {
  description = "Cognito CLI app client ID"
  value       = module.platform.cognito_cli_client_id
}

output "cognito_domain" {
  description = "Cognito custom domain FQDN"
  value       = module.platform.cognito_domain
}

output "cognito_token_endpoint" {
  description = "Cognito OAuth2 token endpoint"
  value       = module.platform.cognito_token_endpoint
}

output "cognito_authorize_endpoint" {
  description = "Cognito OAuth2 authorize endpoint"
  value       = module.platform.cognito_authorize_endpoint
}

output "cognito_logout_endpoint" {
  description = "Cognito logout endpoint"
  value       = module.platform.cognito_logout_endpoint
}

#------------------------------------------------------------------------------
# IAM Identity Center Outputs
#------------------------------------------------------------------------------

output "idc_application_arn" {
  description = "Identity Center application ARN"
  value       = module.platform.idc_application_arn
}

output "idc_instance_id" {
  description = "Identity Center instance ID (e.g. ssoins-abc123def456)"
  value       = module.platform.idc_instance_id
}

output "idc_issuer_url" {
  description = "Identity Center OIDC issuer URL"
  value       = module.platform.idc_issuer_url
}

output "idc_jwks_uri" {
  description = "Identity Center JWKS URI for JWT validation"
  value       = module.platform.idc_jwks_uri
}

output "idc_token_endpoint" {
  description = "Identity Center OAuth2 token endpoint"
  value       = module.platform.idc_token_endpoint
}

output "idc_authorize_endpoint" {
  description = "Identity Center OAuth2 authorize endpoint"
  value       = module.platform.idc_authorize_endpoint
}

output "idc_client_id" {
  description = "Identity Center OAuth2 client ID (same as application ARN)"
  value       = module.platform.idc_client_id
}

output "idc_client_secret" {
  description = "Identity Center OAuth2 client secret (from console)"
  value       = module.platform.idc_client_secret
  sensitive   = true
}

output "idc_admin_email" {
  description = "Identity Center admin user email"
  value       = var.deploy_identity_center ? var.idc_admin_email : null
}

#------------------------------------------------------------------------------
# Keycloak Outputs
#------------------------------------------------------------------------------

output "keycloak_issuer_url" {
  description = "Keycloak OIDC issuer URL"
  value       = module.platform.keycloak_issuer_url
}

output "keycloak_jwks_uri" {
  description = "Keycloak JWKS URI for JWT validation"
  value       = module.platform.keycloak_jwks_uri
}

output "keycloak_token_endpoint" {
  description = "Keycloak OAuth2 token endpoint"
  value       = module.platform.keycloak_token_endpoint
}

output "keycloak_authorize_endpoint" {
  description = "Keycloak OAuth2 authorize endpoint"
  value       = module.platform.keycloak_authorize_endpoint
}

output "keycloak_device_endpoint" {
  description = "Keycloak OAuth2 device authorization endpoint"
  value       = module.platform.keycloak_device_endpoint
}

output "keycloak_logout_endpoint" {
  description = "Keycloak logout endpoint"
  value       = module.platform.keycloak_logout_endpoint
}

output "keycloak_browser_client_id" {
  description = "Keycloak browser-flow client ID"
  value       = module.platform.keycloak_browser_client_id
}

output "keycloak_device_client_id" {
  description = "Keycloak device-flow (CLI) client ID"
  value       = module.platform.keycloak_device_client_id
}

output "keycloak_admin_username" {
  description = "Keycloak admin username for OSMO bootstrap"
  value       = var.deploy_keycloak ? var.keycloak_admin_username : null
}
