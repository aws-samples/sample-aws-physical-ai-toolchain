<!-- BEGIN_TF_DOCS -->


## Requirements

## Requirements

| Name | Version |
|------|---------|
| <a name="requirement_terraform"></a> [terraform](#requirement\_terraform) | >= 1.5.0 |
| <a name="requirement_aws"></a> [aws](#requirement\_aws) | >= 5.0 |
| <a name="requirement_random"></a> [random](#requirement\_random) | >= 3.0 |

## Providers

## Providers

| Name | Version |
|------|---------|
| <a name="provider_aws"></a> [aws](#provider\_aws) | >= 5.0 |
| <a name="provider_aws.us_east_1"></a> [aws.us\_east\_1](#provider\_aws.us\_east\_1) | >= 5.0 |
| <a name="provider_random"></a> [random](#provider\_random) | >= 3.0 |
| <a name="provider_terraform"></a> [terraform](#provider\_terraform) | n/a |

## Modules

## Modules

| Name | Source | Version |
|------|--------|---------|
| <a name="module_rds"></a> [rds](#module\_rds) | terraform-aws-modules/rds/aws | ~> 6.0 |
| <a name="module_vpc"></a> [vpc](#module\_vpc) | terraform-aws-modules/vpc/aws | ~> 5.0 |
| <a name="module_vpc_endpoints"></a> [vpc\_endpoints](#module\_vpc\_endpoints) | terraform-aws-modules/vpc/aws//modules/vpc-endpoints | ~> 5.0 |

## Resources

## Resources

| Name | Type |
|------|------|
| [aws_acm_certificate.osmo](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/acm_certificate) | resource |
| [aws_acm_certificate.osmo_auth](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/acm_certificate) | resource |
| [aws_acm_certificate.osmo_keycloak](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/acm_certificate) | resource |
| [aws_acm_certificate_validation.osmo](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/acm_certificate_validation) | resource |
| [aws_acm_certificate_validation.osmo_auth](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/acm_certificate_validation) | resource |
| [aws_acm_certificate_validation.osmo_keycloak](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/acm_certificate_validation) | resource |
| [aws_cloudwatch_log_group.flow_logs](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/cloudwatch_log_group) | resource |
| [aws_cloudwatch_log_group.waf](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/cloudwatch_log_group) | resource |
| [aws_cognito_user.admin](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/cognito_user) | resource |
| [aws_cognito_user_group.admin](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/cognito_user_group) | resource |
| [aws_cognito_user_group.user](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/cognito_user_group) | resource |
| [aws_cognito_user_in_group.admin_group](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/cognito_user_in_group) | resource |
| [aws_cognito_user_pool.osmo](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/cognito_user_pool) | resource |
| [aws_cognito_user_pool_client.browser](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/cognito_user_pool_client) | resource |
| [aws_cognito_user_pool_client.cli](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/cognito_user_pool_client) | resource |
| [aws_cognito_user_pool_domain.osmo](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/cognito_user_pool_domain) | resource |
| [aws_elasticache_replication_group.redis](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/elasticache_replication_group) | resource |
| [aws_flow_log.vpc](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/flow_log) | resource |
| [aws_guardduty_detector.main](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/guardduty_detector) | resource |
| [aws_iam_access_key.osmo_s3](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_access_key) | resource |
| [aws_iam_policy.osmo_cloudwatch_logs](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_policy) | resource |
| [aws_iam_policy.osmo_ecr_access](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_policy) | resource |
| [aws_iam_policy.osmo_s3_access](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_policy) | resource |
| [aws_iam_policy.osmo_secrets_read](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_policy) | resource |
| [aws_iam_role.flow_logs](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role) | resource |
| [aws_iam_role_policy.flow_logs](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_role_policy) | resource |
| [aws_iam_user.osmo_s3](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_user) | resource |
| [aws_iam_user_policy_attachment.osmo_s3](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_user_policy_attachment) | resource |
| [aws_kms_alias.osmo](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/kms_alias) | resource |
| [aws_kms_key.osmo](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/kms_key) | resource |
| [aws_route53_record.cognito_custom_domain](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/route53_record) | resource |
| [aws_route53_record.cognito_parent_domain](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/route53_record) | resource |
| [aws_route53_record.osmo_auth_cert_validation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/route53_record) | resource |
| [aws_route53_record.osmo_cert_validation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/route53_record) | resource |
| [aws_route53_record.osmo_keycloak_cert_validation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/route53_record) | resource |
| [aws_s3_bucket.datasets](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/s3_bucket) | resource |
| [aws_s3_bucket.workflows](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/s3_bucket) | resource |
| [aws_s3_bucket_cors_configuration.datasets](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/s3_bucket_cors_configuration) | resource |
| [aws_s3_bucket_lifecycle_configuration.datasets](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/s3_bucket_lifecycle_configuration) | resource |
| [aws_s3_bucket_lifecycle_configuration.workflows](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/s3_bucket_lifecycle_configuration) | resource |
| [aws_s3_bucket_public_access_block.datasets](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/s3_bucket_public_access_block) | resource |
| [aws_s3_bucket_public_access_block.workflows](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/s3_bucket_public_access_block) | resource |
| [aws_s3_bucket_server_side_encryption_configuration.datasets](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/s3_bucket_server_side_encryption_configuration) | resource |
| [aws_s3_bucket_server_side_encryption_configuration.workflows](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/s3_bucket_server_side_encryption_configuration) | resource |
| [aws_s3_bucket_versioning.datasets](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/s3_bucket_versioning) | resource |
| [aws_s3_bucket_versioning.workflows](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/s3_bucket_versioning) | resource |
| [aws_secretsmanager_secret.ngc_api_key](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/secretsmanager_secret) | resource |
| [aws_secretsmanager_secret.osmo](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/secretsmanager_secret) | resource |
| [aws_secretsmanager_secret.osmo_s3_credentials](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/secretsmanager_secret) | resource |
| [aws_secretsmanager_secret.rds_password](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/secretsmanager_secret) | resource |
| [aws_secretsmanager_secret.redis_auth_token](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/secretsmanager_secret) | resource |
| [aws_secretsmanager_secret_version.ngc_api_key](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/secretsmanager_secret_version) | resource |
| [aws_secretsmanager_secret_version.osmo](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/secretsmanager_secret_version) | resource |
| [aws_secretsmanager_secret_version.osmo_s3_credentials](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/secretsmanager_secret_version) | resource |
| [aws_secretsmanager_secret_version.rds_password](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/secretsmanager_secret_version) | resource |
| [aws_secretsmanager_secret_version.redis_auth_token](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/secretsmanager_secret_version) | resource |
| [aws_security_group.alb](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/security_group) | resource |
| [aws_security_group.rds](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/security_group) | resource |
| [aws_security_group.redis](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/security_group) | resource |
| [aws_security_group.vpc_endpoints](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/security_group) | resource |
| [aws_security_group_rule.alb_egress_all](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/security_group_rule) | resource |
| [aws_security_group_rule.alb_ingress_https](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/security_group_rule) | resource |
| [aws_security_group_rule.alb_ingress_nat](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/security_group_rule) | resource |
| [aws_security_group_rule.rds_egress](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/security_group_rule) | resource |
| [aws_security_group_rule.rds_from_vpc](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/security_group_rule) | resource |
| [aws_security_group_rule.redis_egress](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/security_group_rule) | resource |
| [aws_security_group_rule.redis_from_vpc](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/security_group_rule) | resource |
| [aws_securityhub_account.main](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/securityhub_account) | resource |
| [aws_securityhub_standards_subscription.aws_foundational](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/securityhub_standards_subscription) | resource |
| [aws_securityhub_standards_subscription.cis](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/securityhub_standards_subscription) | resource |
| [aws_ssoadmin_application.osmo](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/ssoadmin_application) | resource |
| [aws_ssoadmin_application_access_scope.email](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/ssoadmin_application_access_scope) | resource |
| [aws_ssoadmin_application_access_scope.openid](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/ssoadmin_application_access_scope) | resource |
| [aws_ssoadmin_application_access_scope.profile](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/ssoadmin_application_access_scope) | resource |
| [aws_wafv2_ip_set.allowed_ips](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/wafv2_ip_set) | resource |
| [aws_wafv2_web_acl.osmo](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/wafv2_web_acl) | resource |
| [aws_wafv2_web_acl_logging_configuration.osmo](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/wafv2_web_acl_logging_configuration) | resource |
| [random_password.rds_password](https://registry.terraform.io/providers/hashicorp/random/latest/docs/resources/password) | resource |
| [random_password.redis_auth_token](https://registry.terraform.io/providers/hashicorp/random/latest/docs/resources/password) | resource |
| [terraform_data.idc_auth_method](https://registry.terraform.io/providers/hashicorp/terraform/latest/docs/resources/data) | resource |
| [terraform_data.idc_grant](https://registry.terraform.io/providers/hashicorp/terraform/latest/docs/resources/data) | resource |
| [aws_caller_identity.current](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/data-sources/caller_identity) | data source |
| [aws_route53_zone.cognito](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/data-sources/route53_zone) | data source |
| [aws_ssoadmin_instances.this](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/data-sources/ssoadmin_instances) | data source |

## Inputs

## Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|:--------:|
| <a name="input_aws_region"></a> [aws\_region](#input\_aws\_region) | AWS region | `string` | n/a | yes |
| <a name="input_azs"></a> [azs](#input\_azs) | List of availability zones | `list(string)` | n/a | yes |
| <a name="input_cluster_name"></a> [cluster\_name](#input\_cluster\_name) | EKS cluster name for tagging | `string` | n/a | yes |
| <a name="input_common_tags"></a> [common\_tags](#input\_common\_tags) | Common tags to apply to all resources | `map(string)` | n/a | yes |
| <a name="input_database_subnets"></a> [database\_subnets](#input\_database\_subnets) | Database subnet CIDRs | `list(string)` | n/a | yes |
| <a name="input_deploy_postgresql"></a> [deploy\_postgresql](#input\_deploy\_postgresql) | Deploy RDS PostgreSQL | `bool` | n/a | yes |
| <a name="input_deploy_redis"></a> [deploy\_redis](#input\_deploy\_redis) | Deploy ElastiCache Redis | `bool` | n/a | yes |
| <a name="input_elasticache_subnets"></a> [elasticache\_subnets](#input\_elasticache\_subnets) | ElastiCache subnet CIDRs | `list(string)` | n/a | yes |
| <a name="input_environment"></a> [environment](#input\_environment) | Environment name | `string` | n/a | yes |
| <a name="input_kms_enable_key_rotation"></a> [kms\_enable\_key\_rotation](#input\_kms\_enable\_key\_rotation) | Enable KMS key rotation | `bool` | n/a | yes |
| <a name="input_kms_key_deletion_window_days"></a> [kms\_key\_deletion\_window\_days](#input\_kms\_key\_deletion\_window\_days) | KMS key deletion window | `number` | n/a | yes |
| <a name="input_name_prefix"></a> [name\_prefix](#input\_name\_prefix) | Prefix for resource names | `string` | n/a | yes |
| <a name="input_osmo_auth_hostname"></a> [osmo\_auth\_hostname](#input\_osmo\_auth\_hostname) | FQDN for OSMO auth/Keycloak (e.g., osmo-aws-auth.example.com) | `string` | n/a | yes |
| <a name="input_osmo_hostname"></a> [osmo\_hostname](#input\_osmo\_hostname) | FQDN for OSMO service (e.g., osmo-aws.example.com) | `string` | n/a | yes |
| <a name="input_private_subnets"></a> [private\_subnets](#input\_private\_subnets) | Private subnet CIDRs | `list(string)` | n/a | yes |
| <a name="input_public_subnets"></a> [public\_subnets](#input\_public\_subnets) | Public subnet CIDRs | `list(string)` | n/a | yes |
| <a name="input_rds_allocated_storage"></a> [rds\_allocated\_storage](#input\_rds\_allocated\_storage) | Allocated storage in GB | `number` | n/a | yes |
| <a name="input_rds_backup_retention_period"></a> [rds\_backup\_retention\_period](#input\_rds\_backup\_retention\_period) | Backup retention period | `number` | n/a | yes |
| <a name="input_rds_db_name"></a> [rds\_db\_name](#input\_rds\_db\_name) | Database name | `string` | n/a | yes |
| <a name="input_rds_deletion_protection"></a> [rds\_deletion\_protection](#input\_rds\_deletion\_protection) | Enable deletion protection | `bool` | n/a | yes |
| <a name="input_rds_engine_version"></a> [rds\_engine\_version](#input\_rds\_engine\_version) | PostgreSQL engine version | `string` | n/a | yes |
| <a name="input_rds_instance_class"></a> [rds\_instance\_class](#input\_rds\_instance\_class) | RDS instance class | `string` | n/a | yes |
| <a name="input_rds_max_allocated_storage"></a> [rds\_max\_allocated\_storage](#input\_rds\_max\_allocated\_storage) | Maximum allocated storage in GB | `number` | n/a | yes |
| <a name="input_rds_multi_az"></a> [rds\_multi\_az](#input\_rds\_multi\_az) | Enable Multi-AZ | `bool` | n/a | yes |
| <a name="input_rds_username"></a> [rds\_username](#input\_rds\_username) | Master username | `string` | n/a | yes |
| <a name="input_redis_automatic_failover_enabled"></a> [redis\_automatic\_failover\_enabled](#input\_redis\_automatic\_failover\_enabled) | Enable automatic failover | `bool` | n/a | yes |
| <a name="input_redis_engine_version"></a> [redis\_engine\_version](#input\_redis\_engine\_version) | Redis engine version | `string` | n/a | yes |
| <a name="input_redis_multi_az_enabled"></a> [redis\_multi\_az\_enabled](#input\_redis\_multi\_az\_enabled) | Enable Multi-AZ | `bool` | n/a | yes |
| <a name="input_redis_node_type"></a> [redis\_node\_type](#input\_redis\_node\_type) | ElastiCache node type | `string` | n/a | yes |
| <a name="input_redis_num_cache_clusters"></a> [redis\_num\_cache\_clusters](#input\_redis\_num\_cache\_clusters) | Number of cache clusters | `number` | n/a | yes |
| <a name="input_redis_snapshot_retention_limit"></a> [redis\_snapshot\_retention\_limit](#input\_redis\_snapshot\_retention\_limit) | Snapshot retention limit | `number` | n/a | yes |
| <a name="input_route53_zone_id"></a> [route53\_zone\_id](#input\_route53\_zone\_id) | ID of existing Route53 hosted zone | `string` | n/a | yes |
| <a name="input_s3_datasets_bucket"></a> [s3\_datasets\_bucket](#input\_s3\_datasets\_bucket) | S3 bucket name for datasets | `string` | n/a | yes |
| <a name="input_s3_force_destroy"></a> [s3\_force\_destroy](#input\_s3\_force\_destroy) | Allow bucket deletion with objects | `bool` | n/a | yes |
| <a name="input_s3_versioning_enabled"></a> [s3\_versioning\_enabled](#input\_s3\_versioning\_enabled) | Enable bucket versioning | `bool` | n/a | yes |
| <a name="input_s3_workflows_bucket"></a> [s3\_workflows\_bucket](#input\_s3\_workflows\_bucket) | S3 bucket name for workflows | `string` | n/a | yes |
| <a name="input_secrets_recovery_window_days"></a> [secrets\_recovery\_window\_days](#input\_secrets\_recovery\_window\_days) | Secrets Manager recovery window | `number` | n/a | yes |
| <a name="input_single_nat_gateway"></a> [single\_nat\_gateway](#input\_single\_nat\_gateway) | Use single NAT gateway | `bool` | n/a | yes |
| <a name="input_vpc_cidr"></a> [vpc\_cidr](#input\_vpc\_cidr) | CIDR block for VPC | `string` | n/a | yes |
| <a name="input_alb_allowed_cidrs"></a> [alb\_allowed\_cidrs](#input\_alb\_allowed\_cidrs) | CIDRs allowed to access the ALB on port 443. Empty = no SG created. | `list(string)` | `[]` | no |
| <a name="input_cognito_admin_email"></a> [cognito\_admin\_email](#input\_cognito\_admin\_email) | Email for the initial Cognito admin user | `string` | `""` | no |
| <a name="input_cognito_admin_temp_password"></a> [cognito\_admin\_temp\_password](#input\_cognito\_admin\_temp\_password) | Temporary password for the initial Cognito admin user (user must change on first login) | `string` | `"ChangeMe123!"` | no |
| <a name="input_cognito_admin_username"></a> [cognito\_admin\_username](#input\_cognito\_admin\_username) | Username for the initial Cognito admin user | `string` | `"osmo-admin"` | no |
| <a name="input_cognito_parent_domain_placeholder_ip"></a> [cognito\_parent\_domain\_placeholder\_ip](#input\_cognito\_parent\_domain\_placeholder\_ip) | If set, create an A record for the parent domain of osmo\_auth\_hostname so Cognito custom domain validation passes (e.g. 192.0.2.1). Leave empty if parent domain already has an A record. | `string` | `"192.0.2.1"` | no |
| <a name="input_deploy_cognito"></a> [deploy\_cognito](#input\_deploy\_cognito) | Deploy AWS Cognito User Pool as the OSMO identity provider | `bool` | `true` | no |
| <a name="input_deploy_identity_center"></a> [deploy\_identity\_center](#input\_deploy\_identity\_center) | Create an IAM Identity Center OAuth 2.0 application for OSMO | `bool` | `false` | no |
| <a name="input_deploy_keycloak"></a> [deploy\_keycloak](#input\_deploy\_keycloak) | Provision ACM certificate and outputs for self-hosted Keycloak | `bool` | `true` | no |
| <a name="input_enable_flow_logs"></a> [enable\_flow\_logs](#input\_enable\_flow\_logs) | Enable VPC Flow Logs | `bool` | `false` | no |
| <a name="input_enable_guardduty"></a> [enable\_guardduty](#input\_enable\_guardduty) | Enable AWS GuardDuty | `bool` | `false` | no |
| <a name="input_enable_security_hub"></a> [enable\_security\_hub](#input\_enable\_security\_hub) | Enable AWS Security Hub | `bool` | `false` | no |
| <a name="input_enable_waf"></a> [enable\_waf](#input\_enable\_waf) | Enable AWS WAF for ALB protection | `bool` | `false` | no |
| <a name="input_flow_logs_retention_days"></a> [flow\_logs\_retention\_days](#input\_flow\_logs\_retention\_days) | Retention period for VPC Flow Logs | `number` | `30` | no |
| <a name="input_idc_client_secret"></a> [idc\_client\_secret](#input\_idc\_client\_secret) | OAuth2 client secret — generated in the Identity Center console after terraform apply | `string` | `""` | no |
| <a name="input_idc_region"></a> [idc\_region](#input\_idc\_region) | AWS region where Identity Center is enabled (defaults to aws\_region) | `string` | `""` | no |
| <a name="input_keycloak_realm"></a> [keycloak\_realm](#input\_keycloak\_realm) | Keycloak realm name for OSMO | `string` | `"osmo"` | no |
| <a name="input_restrict_rds_to_eks_only"></a> [restrict\_rds\_to\_eks\_only](#input\_restrict\_rds\_to\_eks\_only) | Restrict RDS access to EKS security group only | `bool` | `false` | no |
| <a name="input_restrict_redis_to_eks_only"></a> [restrict\_redis\_to\_eks\_only](#input\_restrict\_redis\_to\_eks\_only) | Restrict Redis access to EKS security group only | `bool` | `false` | no |
| <a name="input_s3_allowed_origins"></a> [s3\_allowed\_origins](#input\_s3\_allowed\_origins) | Allowed origins for S3 CORS | `list(string)` | `[]` | no |
| <a name="input_waf_allowed_ip_cidrs"></a> [waf\_allowed\_ip\_cidrs](#input\_waf\_allowed\_ip\_cidrs) | List of CIDR blocks allowed to access the ALB | `list(string)` | `[]` | no |
| <a name="input_waf_block_mode"></a> [waf\_block\_mode](#input\_waf\_block\_mode) | WAF action mode: COUNT or BLOCK | `string` | `"BLOCK"` | no |
| <a name="input_waf_rate_limit"></a> [waf\_rate\_limit](#input\_waf\_rate\_limit) | Rate limit for requests per 5-minute period per IP | `number` | `2000` | no |

## Outputs

## Outputs

| Name | Description |
|------|-------------|
| <a name="output_acm_auth_certificate_arn"></a> [acm\_auth\_certificate\_arn](#output\_acm\_auth\_certificate\_arn) | ACM certificate ARN for OSMO auth domain |
| <a name="output_acm_certificate_arn"></a> [acm\_certificate\_arn](#output\_acm\_certificate\_arn) | ACM certificate ARN for OSMO service domain |
| <a name="output_alb_security_group_id"></a> [alb\_security\_group\_id](#output\_alb\_security\_group\_id) | ALB security group ID (null when alb\_allowed\_cidrs is empty) |
| <a name="output_cognito_authorize_endpoint"></a> [cognito\_authorize\_endpoint](#output\_cognito\_authorize\_endpoint) | Cognito OAuth2 authorize endpoint |
| <a name="output_cognito_browser_client_id"></a> [cognito\_browser\_client\_id](#output\_cognito\_browser\_client\_id) | Cognito browser app client ID |
| <a name="output_cognito_browser_client_secret"></a> [cognito\_browser\_client\_secret](#output\_cognito\_browser\_client\_secret) | Cognito browser app client secret |
| <a name="output_cognito_cli_client_id"></a> [cognito\_cli\_client\_id](#output\_cognito\_cli\_client\_id) | Cognito CLI app client ID |
| <a name="output_cognito_domain"></a> [cognito\_domain](#output\_cognito\_domain) | Cognito custom domain FQDN |
| <a name="output_cognito_issuer_url"></a> [cognito\_issuer\_url](#output\_cognito\_issuer\_url) | Cognito OIDC issuer URL |
| <a name="output_cognito_jwks_uri"></a> [cognito\_jwks\_uri](#output\_cognito\_jwks\_uri) | Cognito JWKS URI for JWT validation |
| <a name="output_cognito_logout_endpoint"></a> [cognito\_logout\_endpoint](#output\_cognito\_logout\_endpoint) | Cognito logout endpoint |
| <a name="output_cognito_token_endpoint"></a> [cognito\_token\_endpoint](#output\_cognito\_token\_endpoint) | Cognito OAuth2 token endpoint |
| <a name="output_cognito_user_pool_arn"></a> [cognito\_user\_pool\_arn](#output\_cognito\_user\_pool\_arn) | Cognito User Pool ARN |
| <a name="output_cognito_user_pool_id"></a> [cognito\_user\_pool\_id](#output\_cognito\_user\_pool\_id) | Cognito User Pool ID |
| <a name="output_database_subnet_group_name"></a> [database\_subnet\_group\_name](#output\_database\_subnet\_group\_name) | Database subnet group name |
| <a name="output_database_subnet_ids"></a> [database\_subnet\_ids](#output\_database\_subnet\_ids) | List of database subnet IDs |
| <a name="output_elasticache_subnet_group_name"></a> [elasticache\_subnet\_group\_name](#output\_elasticache\_subnet\_group\_name) | ElastiCache subnet group name |
| <a name="output_elasticache_subnet_ids"></a> [elasticache\_subnet\_ids](#output\_elasticache\_subnet\_ids) | List of elasticache subnet IDs |
| <a name="output_flow_logs_log_group_arn"></a> [flow\_logs\_log\_group\_arn](#output\_flow\_logs\_log\_group\_arn) | VPC Flow Logs CloudWatch Log Group ARN |
| <a name="output_guardduty_detector_id"></a> [guardduty\_detector\_id](#output\_guardduty\_detector\_id) | GuardDuty detector ID |
| <a name="output_idc_application_arn"></a> [idc\_application\_arn](#output\_idc\_application\_arn) | Identity Center application ARN (also the client ID) |
| <a name="output_idc_authorize_endpoint"></a> [idc\_authorize\_endpoint](#output\_idc\_authorize\_endpoint) | Identity Center authorize endpoint |
| <a name="output_idc_client_id"></a> [idc\_client\_id](#output\_idc\_client\_id) | Identity Center OAuth2 client ID (same as application ARN) |
| <a name="output_idc_client_secret"></a> [idc\_client\_secret](#output\_idc\_client\_secret) | Identity Center OAuth2 client secret (from console) |
| <a name="output_idc_instance_id"></a> [idc\_instance\_id](#output\_idc\_instance\_id) | Identity Center instance ID (e.g. ssoins-abc123def456) |
| <a name="output_idc_issuer_url"></a> [idc\_issuer\_url](#output\_idc\_issuer\_url) | Identity Center OIDC issuer URL |
| <a name="output_idc_jwks_uri"></a> [idc\_jwks\_uri](#output\_idc\_jwks\_uri) | Identity Center JWKS URI |
| <a name="output_idc_token_endpoint"></a> [idc\_token\_endpoint](#output\_idc\_token\_endpoint) | Identity Center token endpoint |
| <a name="output_keycloak_authorize_endpoint"></a> [keycloak\_authorize\_endpoint](#output\_keycloak\_authorize\_endpoint) | Keycloak OAuth2 authorize endpoint |
| <a name="output_keycloak_browser_client_id"></a> [keycloak\_browser\_client\_id](#output\_keycloak\_browser\_client\_id) | Keycloak browser-flow client ID |
| <a name="output_keycloak_device_client_id"></a> [keycloak\_device\_client\_id](#output\_keycloak\_device\_client\_id) | Keycloak device-flow (CLI) client ID |
| <a name="output_keycloak_device_endpoint"></a> [keycloak\_device\_endpoint](#output\_keycloak\_device\_endpoint) | Keycloak OAuth2 device authorization endpoint |
| <a name="output_keycloak_issuer_url"></a> [keycloak\_issuer\_url](#output\_keycloak\_issuer\_url) | Keycloak OIDC issuer URL |
| <a name="output_keycloak_jwks_uri"></a> [keycloak\_jwks\_uri](#output\_keycloak\_jwks\_uri) | Keycloak JWKS URI for JWT validation |
| <a name="output_keycloak_logout_endpoint"></a> [keycloak\_logout\_endpoint](#output\_keycloak\_logout\_endpoint) | Keycloak logout endpoint |
| <a name="output_keycloak_token_endpoint"></a> [keycloak\_token\_endpoint](#output\_keycloak\_token\_endpoint) | Keycloak OAuth2 token endpoint |
| <a name="output_kms_key_alias"></a> [kms\_key\_alias](#output\_kms\_key\_alias) | KMS key alias |
| <a name="output_kms_key_arn"></a> [kms\_key\_arn](#output\_kms\_key\_arn) | KMS key ARN |
| <a name="output_kms_key_id"></a> [kms\_key\_id](#output\_kms\_key\_id) | KMS key ID |
| <a name="output_nat_gateway_ids"></a> [nat\_gateway\_ids](#output\_nat\_gateway\_ids) | NAT Gateway IDs |
| <a name="output_ngc_api_key_secret_arn"></a> [ngc\_api\_key\_secret\_arn](#output\_ngc\_api\_key\_secret\_arn) | NGC API key secret ARN |
| <a name="output_osmo_cloudwatch_logs_policy_arn"></a> [osmo\_cloudwatch\_logs\_policy\_arn](#output\_osmo\_cloudwatch\_logs\_policy\_arn) | OSMO CloudWatch Logs policy ARN |
| <a name="output_osmo_ecr_access_policy_arn"></a> [osmo\_ecr\_access\_policy\_arn](#output\_osmo\_ecr\_access\_policy\_arn) | OSMO ECR access policy ARN |
| <a name="output_osmo_s3_access_policy_arn"></a> [osmo\_s3\_access\_policy\_arn](#output\_osmo\_s3\_access\_policy\_arn) | OSMO S3 access policy ARN |
| <a name="output_osmo_s3_credentials_secret_arn"></a> [osmo\_s3\_credentials\_secret\_arn](#output\_osmo\_s3\_credentials\_secret\_arn) | ARN of the Secrets Manager secret containing OSMO S3 IAM user credentials |
| <a name="output_osmo_secrets_read_policy_arn"></a> [osmo\_secrets\_read\_policy\_arn](#output\_osmo\_secrets\_read\_policy\_arn) | OSMO secrets read policy ARN |
| <a name="output_private_subnet_ids"></a> [private\_subnet\_ids](#output\_private\_subnet\_ids) | List of private subnet IDs |
| <a name="output_public_subnet_ids"></a> [public\_subnet\_ids](#output\_public\_subnet\_ids) | List of public subnet IDs |
| <a name="output_rds_database_name"></a> [rds\_database\_name](#output\_rds\_database\_name) | RDS database name |
| <a name="output_rds_endpoint"></a> [rds\_endpoint](#output\_rds\_endpoint) | RDS endpoint |
| <a name="output_rds_password_secret_arn"></a> [rds\_password\_secret\_arn](#output\_rds\_password\_secret\_arn) | ARN of the Secrets Manager secret containing RDS password |
| <a name="output_rds_port"></a> [rds\_port](#output\_rds\_port) | RDS port |
| <a name="output_rds_security_group_id"></a> [rds\_security\_group\_id](#output\_rds\_security\_group\_id) | RDS security group ID |
| <a name="output_rds_username"></a> [rds\_username](#output\_rds\_username) | RDS master username |
| <a name="output_redis_auth_token_secret_arn"></a> [redis\_auth\_token\_secret\_arn](#output\_redis\_auth\_token\_secret\_arn) | ARN of the Secrets Manager secret containing Redis auth token |
| <a name="output_redis_endpoint"></a> [redis\_endpoint](#output\_redis\_endpoint) | Redis primary endpoint |
| <a name="output_redis_port"></a> [redis\_port](#output\_redis\_port) | Redis port |
| <a name="output_redis_security_group_id"></a> [redis\_security\_group\_id](#output\_redis\_security\_group\_id) | Redis security group ID |
| <a name="output_s3_datasets_bucket_arn"></a> [s3\_datasets\_bucket\_arn](#output\_s3\_datasets\_bucket\_arn) | S3 datasets bucket ARN |
| <a name="output_s3_datasets_bucket_name"></a> [s3\_datasets\_bucket\_name](#output\_s3\_datasets\_bucket\_name) | S3 datasets bucket name |
| <a name="output_s3_workflows_bucket_arn"></a> [s3\_workflows\_bucket\_arn](#output\_s3\_workflows\_bucket\_arn) | S3 workflows bucket ARN |
| <a name="output_s3_workflows_bucket_name"></a> [s3\_workflows\_bucket\_name](#output\_s3\_workflows\_bucket\_name) | S3 workflows bucket name |
| <a name="output_secrets_manager_arn"></a> [secrets\_manager\_arn](#output\_secrets\_manager\_arn) | OSMO secrets ARN |
| <a name="output_vpc_cidr_block"></a> [vpc\_cidr\_block](#output\_vpc\_cidr\_block) | VPC CIDR block |
| <a name="output_vpc_id"></a> [vpc\_id](#output\_vpc\_id) | VPC ID |
| <a name="output_waf_web_acl_arn"></a> [waf\_web\_acl\_arn](#output\_waf\_web\_acl\_arn) | WAF Web ACL ARN |
| <a name="output_waf_web_acl_id"></a> [waf\_web\_acl\_id](#output\_waf\_web\_acl\_id) | WAF Web ACL ID |

<!-- END_TF_DOCS -->