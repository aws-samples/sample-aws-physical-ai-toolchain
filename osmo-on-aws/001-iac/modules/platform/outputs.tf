# SPDX-License-Identifier: Apache-2.0

#------------------------------------------------------------------------------
# VPC Outputs
#------------------------------------------------------------------------------

output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "VPC CIDR block"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnet_ids" {
  description = "List of private subnet IDs"
  value       = module.vpc.private_subnets
}

output "public_subnet_ids" {
  description = "List of public subnet IDs"
  value       = module.vpc.public_subnets
}

output "database_subnet_ids" {
  description = "List of database subnet IDs"
  value       = module.vpc.database_subnets
}

output "elasticache_subnet_ids" {
  description = "List of elasticache subnet IDs"
  value       = module.vpc.elasticache_subnets
}

output "database_subnet_group_name" {
  description = "Database subnet group name"
  value       = module.vpc.database_subnet_group_name
}

output "elasticache_subnet_group_name" {
  description = "ElastiCache subnet group name"
  value       = module.vpc.elasticache_subnet_group_name
}

output "nat_gateway_ids" {
  description = "NAT Gateway IDs"
  value       = module.vpc.natgw_ids
}

#------------------------------------------------------------------------------
# RDS Outputs
#------------------------------------------------------------------------------

output "rds_endpoint" {
  description = "RDS endpoint"
  value       = var.deploy_postgresql ? module.rds[0].db_instance_endpoint : null
}

output "rds_port" {
  description = "RDS port"
  value       = var.deploy_postgresql ? module.rds[0].db_instance_port : null
}

output "rds_database_name" {
  description = "RDS database name"
  value       = var.deploy_postgresql ? module.rds[0].db_instance_name : null
}

output "rds_username" {
  description = "RDS master username"
  value       = var.deploy_postgresql ? module.rds[0].db_instance_username : null
  sensitive   = true
}

output "rds_password_secret_arn" {
  description = "ARN of the Secrets Manager secret containing RDS password"
  value       = var.deploy_postgresql ? aws_secretsmanager_secret.rds_password[0].arn : null
}

output "rds_security_group_id" {
  description = "RDS security group ID"
  value       = var.deploy_postgresql ? aws_security_group.rds[0].id : null
}

#------------------------------------------------------------------------------
# Redis Outputs
#------------------------------------------------------------------------------

output "redis_endpoint" {
  description = "Redis primary endpoint"
  value       = var.deploy_redis ? aws_elasticache_replication_group.redis[0].primary_endpoint_address : null
}

output "redis_port" {
  description = "Redis port"
  value       = var.deploy_redis ? 6379 : null
}

output "redis_auth_token_secret_arn" {
  description = "ARN of the Secrets Manager secret containing Redis auth token"
  value       = var.deploy_redis ? aws_secretsmanager_secret.redis_auth_token[0].arn : null
}

output "redis_security_group_id" {
  description = "Redis security group ID"
  value       = var.deploy_redis ? aws_security_group.redis[0].id : null
}

#------------------------------------------------------------------------------
# S3 Outputs
#------------------------------------------------------------------------------

output "s3_workflows_bucket_name" {
  description = "S3 workflows bucket name"
  value       = aws_s3_bucket.workflows.id
}

output "s3_workflows_bucket_arn" {
  description = "S3 workflows bucket ARN"
  value       = aws_s3_bucket.workflows.arn
}

output "s3_datasets_bucket_name" {
  description = "S3 datasets bucket name"
  value       = aws_s3_bucket.datasets.id
}

output "s3_datasets_bucket_arn" {
  description = "S3 datasets bucket ARN"
  value       = aws_s3_bucket.datasets.arn
}

#------------------------------------------------------------------------------
# KMS Outputs
#------------------------------------------------------------------------------

output "kms_key_arn" {
  description = "KMS key ARN"
  value       = aws_kms_key.osmo.arn
}

output "kms_key_id" {
  description = "KMS key ID"
  value       = aws_kms_key.osmo.key_id
}

output "kms_key_alias" {
  description = "KMS key alias"
  value       = aws_kms_alias.osmo.name
}

#------------------------------------------------------------------------------
# Secrets Manager Outputs
#------------------------------------------------------------------------------

output "secrets_manager_arn" {
  description = "OSMO secrets ARN"
  value       = aws_secretsmanager_secret.osmo.arn
}

output "ngc_api_key_secret_arn" {
  description = "NGC API key secret ARN"
  value       = aws_secretsmanager_secret.ngc_api_key.arn
}

output "osmo_s3_credentials_secret_arn" {
  description = "ARN of the Secrets Manager secret containing OSMO S3 IAM user credentials"
  value       = aws_secretsmanager_secret.osmo_s3_credentials.arn
}

#------------------------------------------------------------------------------
# IAM Policy Outputs
#------------------------------------------------------------------------------

output "osmo_s3_access_policy_arn" {
  description = "OSMO S3 access policy ARN"
  value       = aws_iam_policy.osmo_s3_access.arn
}

output "osmo_secrets_read_policy_arn" {
  description = "OSMO secrets read policy ARN"
  value       = aws_iam_policy.osmo_secrets_read.arn
}

output "osmo_ecr_access_policy_arn" {
  description = "OSMO ECR access policy ARN"
  value       = aws_iam_policy.osmo_ecr_access.arn
}

output "osmo_cloudwatch_logs_policy_arn" {
  description = "OSMO CloudWatch Logs policy ARN"
  value       = aws_iam_policy.osmo_cloudwatch_logs.arn
}

#------------------------------------------------------------------------------
# ACM / DNS Outputs
#------------------------------------------------------------------------------

output "acm_certificate_arn" {
  description = "ACM certificate ARN for OSMO service domain"
  value       = aws_acm_certificate_validation.osmo.certificate_arn
}

output "acm_auth_certificate_arn" {
  description = "ACM certificate ARN for OSMO auth domain"
  value = (
    var.deploy_cognito ? aws_acm_certificate_validation.osmo_auth[0].certificate_arn :
    var.deploy_keycloak ? aws_acm_certificate_validation.osmo_keycloak[0].certificate_arn :
    null
  )
}

#------------------------------------------------------------------------------
# WAF Outputs
#------------------------------------------------------------------------------

output "waf_web_acl_arn" {
  description = "WAF Web ACL ARN"
  value       = var.enable_waf ? aws_wafv2_web_acl.osmo[0].arn : null
}

output "waf_web_acl_id" {
  description = "WAF Web ACL ID"
  value       = var.enable_waf ? aws_wafv2_web_acl.osmo[0].id : null
}

#------------------------------------------------------------------------------
# Security Outputs
#------------------------------------------------------------------------------

output "flow_logs_log_group_arn" {
  description = "VPC Flow Logs CloudWatch Log Group ARN"
  value       = var.enable_flow_logs ? aws_cloudwatch_log_group.flow_logs[0].arn : null
}

output "guardduty_detector_id" {
  description = "GuardDuty detector ID"
  value       = var.enable_guardduty ? aws_guardduty_detector.main[0].id : null
}

#------------------------------------------------------------------------------
# ALB Security Group Outputs
#------------------------------------------------------------------------------

output "alb_security_group_id" {
  description = "ALB security group ID (null when alb_allowed_cidrs is empty)"
  value       = length(var.alb_allowed_cidrs) > 0 ? aws_security_group.alb[0].id : null
}

#------------------------------------------------------------------------------
# Cognito Outputs
#------------------------------------------------------------------------------

output "cognito_user_pool_id" {
  description = "Cognito User Pool ID"
  value       = var.deploy_cognito ? aws_cognito_user_pool.osmo[0].id : null
}

output "cognito_user_pool_arn" {
  description = "Cognito User Pool ARN"
  value       = var.deploy_cognito ? aws_cognito_user_pool.osmo[0].arn : null
}

output "cognito_issuer_url" {
  description = "Cognito OIDC issuer URL"
  value       = var.deploy_cognito ? "https://cognito-idp.${var.aws_region}.amazonaws.com/${aws_cognito_user_pool.osmo[0].id}" : null
}

output "cognito_jwks_uri" {
  description = "Cognito JWKS URI for JWT validation"
  value       = var.deploy_cognito ? "https://cognito-idp.${var.aws_region}.amazonaws.com/${aws_cognito_user_pool.osmo[0].id}/.well-known/jwks.json" : null
}

output "cognito_browser_client_id" {
  description = "Cognito browser app client ID"
  value       = var.deploy_cognito ? aws_cognito_user_pool_client.browser[0].id : null
}

output "cognito_browser_client_secret" {
  description = "Cognito browser app client secret"
  value       = var.deploy_cognito ? aws_cognito_user_pool_client.browser[0].client_secret : null
  sensitive   = true
}

output "cognito_cli_client_id" {
  description = "Cognito CLI app client ID"
  value       = var.deploy_cognito ? aws_cognito_user_pool_client.cli[0].id : null
}

output "cognito_domain" {
  description = "Cognito custom domain FQDN"
  value       = var.deploy_cognito ? var.osmo_auth_hostname : null
}

output "cognito_token_endpoint" {
  description = "Cognito OAuth2 token endpoint"
  value       = var.deploy_cognito ? "https://${var.osmo_auth_hostname}/oauth2/token" : null
}

output "cognito_authorize_endpoint" {
  description = "Cognito OAuth2 authorize endpoint"
  value       = var.deploy_cognito ? "https://${var.osmo_auth_hostname}/oauth2/authorize" : null
}

output "cognito_logout_endpoint" {
  description = "Cognito logout endpoint"
  value       = var.deploy_cognito ? "https://${var.osmo_auth_hostname}/logout" : null
}

#------------------------------------------------------------------------------
# Identity Center Outputs
#------------------------------------------------------------------------------

output "idc_application_arn" {
  description = "Identity Center application ARN (also the client ID)"
  value       = var.deploy_identity_center ? aws_ssoadmin_application.osmo[0].application_arn : null
}

output "idc_instance_id" {
  description = "Identity Center instance ID (e.g. ssoins-abc123def456)"
  value       = var.deploy_identity_center ? local.idc_instance_id_from_arn : null
}

output "idc_issuer_url" {
  description = "Identity Center OIDC issuer URL"
  value       = var.deploy_identity_center ? "https://identitycenter.${local.idc_region}.amazonaws.com/${local.idc_instance_id_from_arn}" : null
}

output "idc_jwks_uri" {
  description = "Identity Center JWKS URI"
  value       = var.deploy_identity_center ? "https://oidc.${local.idc_region}.amazonaws.com/keys" : null
}

output "idc_token_endpoint" {
  description = "Identity Center token endpoint"
  value       = var.deploy_identity_center ? "https://oidc.${local.idc_region}.amazonaws.com/token" : null
}

output "idc_authorize_endpoint" {
  description = "Identity Center authorize endpoint"
  value       = var.deploy_identity_center ? "https://${local.idc_instance_id_from_arn}.awsapps.com/start/authorize" : null
}

output "idc_client_id" {
  description = "Identity Center OAuth2 client ID (same as application ARN)"
  value       = var.deploy_identity_center ? aws_ssoadmin_application.osmo[0].application_arn : null
}

output "idc_client_secret" {
  description = "Identity Center OAuth2 client secret (from console)"
  value       = var.deploy_identity_center ? var.idc_client_secret : null
  sensitive   = true
}

#------------------------------------------------------------------------------
# Keycloak Outputs (derived from osmo_auth_hostname + realm)
#------------------------------------------------------------------------------

output "keycloak_issuer_url" {
  description = "Keycloak OIDC issuer URL"
  value       = var.deploy_keycloak ? "https://${var.osmo_auth_hostname}/realms/${var.keycloak_realm}" : null
}

output "keycloak_jwks_uri" {
  description = "Keycloak JWKS URI for JWT validation"
  value       = var.deploy_keycloak ? "https://${var.osmo_auth_hostname}/realms/${var.keycloak_realm}/protocol/openid-connect/certs" : null
}

output "keycloak_token_endpoint" {
  description = "Keycloak OAuth2 token endpoint"
  value       = var.deploy_keycloak ? "https://${var.osmo_auth_hostname}/realms/${var.keycloak_realm}/protocol/openid-connect/token" : null
}

output "keycloak_authorize_endpoint" {
  description = "Keycloak OAuth2 authorize endpoint"
  value       = var.deploy_keycloak ? "https://${var.osmo_auth_hostname}/realms/${var.keycloak_realm}/protocol/openid-connect/auth" : null
}

output "keycloak_device_endpoint" {
  description = "Keycloak OAuth2 device authorization endpoint"
  value       = var.deploy_keycloak ? "https://${var.osmo_auth_hostname}/realms/${var.keycloak_realm}/protocol/openid-connect/auth/device" : null
}

output "keycloak_logout_endpoint" {
  description = "Keycloak logout endpoint"
  value       = var.deploy_keycloak ? "https://${var.osmo_auth_hostname}/realms/${var.keycloak_realm}/protocol/openid-connect/logout" : null
}

output "keycloak_browser_client_id" {
  description = "Keycloak browser-flow client ID"
  value       = var.deploy_keycloak ? "osmo-browser-flow" : null
}

output "keycloak_device_client_id" {
  description = "Keycloak device-flow (CLI) client ID"
  value       = var.deploy_keycloak ? "osmo-device" : null
}
