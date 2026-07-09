# OSMO on AWS - Infrastructure as Code

This directory contains Terraform configuration for deploying OSMO infrastructure on AWS.

## Prerequisites

- Terraform >= 1.5.0
- AWS CLI configured with appropriate credentials
- `kubectl` for Kubernetes access after deployment

## Quick Start

1. Copy the example tfvars file:
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   # Or for specific environments:
   cp terraform.tfvars.dev.example terraform.tfvars
   cp terraform.tfvars.prod.example terraform.tfvars
   ```

2. Edit `terraform.tfvars` with your configuration.

3. Initialize and apply:
   ```bash
   terraform init
   terraform plan
   terraform apply
   ```

4. Configure kubectl:
   ```bash
   aws eks update-kubeconfig --region <region> --name <cluster-name>
   ```

## Deployment Modes

| Mode | Description | Components Deployed |
|------|-------------|---------------------|
| `full` | Complete deployment | VPC, EKS, RDS, Redis, S3, GPU nodes |
| `control-plane-only` | Control plane without compute | VPC, EKS, RDS, Redis, S3 (no GPU nodes) |
| `backend-only` | Compute plane only | VPC, EKS, S3, GPU nodes (no RDS/Redis) |

Set the mode in `terraform.tfvars`:
```hcl
deployment_mode = "full"
```

## Module Structure

```
001-iac/
├── main.tf              # Root module composition
├── variables.tf         # Input variables
├── outputs.tf           # Output values
├── locals.tf            # Local values and calculations
├── versions.tf          # Provider requirements
├── backend.tf.example   # S3 backend configuration
└── modules/
    ├── platform/        # Shared AWS services
    │   ├── vpc.tf       # VPC, subnets, NAT gateways
    │   ├── rds.tf       # RDS PostgreSQL
    │   ├── elasticache.tf # ElastiCache Redis
    │   ├── s3.tf        # S3 buckets
    │   ├── kms.tf       # KMS encryption keys
    │   ├── secrets-manager.tf # Secrets Manager
    │   └── iam.tf       # IAM policies
    └── eks/             # EKS cluster
        ├── main.tf      # EKS cluster configuration
        ├── node-groups.tf # Node group definitions
        ├── addons.tf    # EKS addons
        └── irsa.tf      # IRSA roles for services
```

## Key Outputs

After deployment, Terraform outputs include:

- `cluster_name` - EKS cluster name
- `cluster_endpoint` - EKS API endpoint
- `configure_kubectl` - AWS CLI command to configure kubectl
- `rds_endpoint` - RDS PostgreSQL endpoint
- `redis_endpoint` - Redis endpoint
- `s3_workflows_bucket_name` - S3 bucket for workflows
- `s3_datasets_bucket_name` - S3 bucket for datasets
- IRSA role ARNs for service configuration, including:
  - `osmo_service_role_arn` - core OSMO services (SA `osmo-service`)
  - `osmo_backend_role_arn` - backend operator SAs (`osmo-operator-backend-{listener,worker}`)
  - `osmo_workflow_role_arn` - **workflow task pods** (SA `osmo-workflow`); lets `osmo-ctrl`
    do dataset S3 I/O via IRSA instead of the EKS node role

> The IRSA trust policies are aligned to the exact ServiceAccount names the OSMO
> Helm charts create (chart SA name == trust policy subject). The legacy static
> IAM user (`*-osmo-s3`) is retained for compatibility but is no longer used by
> workflows in the 6.3 IRSA model.

## State Management

For production, use S3 backend:

1. Create state bucket:
   ```bash
   aws s3 mb s3://your-terraform-state-bucket
   aws dynamodb create-table \
     --table-name terraform-state-lock \
     --attribute-definitions AttributeName=LockID,AttributeType=S \
     --key-schema AttributeName=LockID,KeyType=HASH \
     --billing-mode PAY_PER_REQUEST
   ```

2. Copy and configure backend:
   ```bash
   cp backend.tf.example backend.tf
   # Edit backend.tf with your bucket name
   ```

## Cleanup

```bash
terraform destroy
```

**Note:** If `rds_deletion_protection = true`, you must first disable it in the AWS console or set it to `false` in tfvars before destroying.

<!-- BEGIN_TF_DOCS -->


## Requirements

## Requirements

| Name | Version |
|------|---------|
| <a name="requirement_terraform"></a> [terraform](#requirement\_terraform) | >= 1.5.0 |
| <a name="requirement_aws"></a> [aws](#requirement\_aws) | ~> 5.0 |
| <a name="requirement_helm"></a> [helm](#requirement\_helm) | ~> 2.12 |
| <a name="requirement_kubernetes"></a> [kubernetes](#requirement\_kubernetes) | ~> 2.25 |
| <a name="requirement_random"></a> [random](#requirement\_random) | ~> 3.6 |
| <a name="requirement_tls"></a> [tls](#requirement\_tls) | ~> 4.0 |

## Providers

## Providers

| Name | Version |
|------|---------|
| <a name="provider_aws"></a> [aws](#provider\_aws) | ~> 5.0 |

## Modules

## Modules

| Name | Source | Version |
|------|--------|---------|
| <a name="module_eks"></a> [eks](#module\_eks) | ./modules/eks | n/a |
| <a name="module_fluentbit_irsa"></a> [fluentbit\_irsa](#module\_fluentbit\_irsa) | terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks | ~> 5.0 |
| <a name="module_platform"></a> [platform](#module\_platform) | ./modules/platform | n/a |

## Resources

## Resources

| Name | Type |
|------|------|
| [aws_cloudwatch_log_group.osmo_logs](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/cloudwatch_log_group) | resource |
| [aws_iam_policy.fluentbit](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_policy) | resource |
| [aws_prometheus_scraper.osmo](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/prometheus_scraper) | resource |
| [aws_prometheus_workspace.osmo](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/prometheus_workspace) | resource |
| [aws_security_group_rule.nodes_from_alb](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/security_group_rule) | resource |
| [aws_security_group_rule.rds_from_eks](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/security_group_rule) | resource |
| [aws_security_group_rule.redis_from_eks](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/security_group_rule) | resource |
| [aws_availability_zones.available](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/data-sources/availability_zones) | data source |

## Inputs

## Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|:--------:|
| <a name="input_aws_region"></a> [aws\_region](#input\_aws\_region) | AWS region for all resources | `string` | n/a | yes |
| <a name="input_cluster_name"></a> [cluster\_name](#input\_cluster\_name) | Name for the EKS cluster and related resources | `string` | n/a | yes |
| <a name="input_resource_suffix"></a> [resource\_suffix](#input\_resource\_suffix) | Suffix for resource names to avoid collisions (e.g. osm01, dev01). Use a unique value per deployment when sharing the same cluster\_name and environment. Required - no default. | `string` | n/a | yes |
| <a name="input_route53_zone_id"></a> [route53\_zone\_id](#input\_route53\_zone\_id) | ID of existing Route53 hosted zone (e.g., ZEXAMPLEZONEID123456) | `string` | n/a | yes |
| <a name="input_route53_zone_name"></a> [route53\_zone\_name](#input\_route53\_zone\_name) | Name of existing Route53 hosted zone (e.g., example.com) | `string` | n/a | yes |
| <a name="input_alb_allowed_cidrs"></a> [alb\_allowed\_cidrs](#input\_alb\_allowed\_cidrs) | CIDRs allowed to access the ALB on port 443 (e.g. your IP: ["1.2.3.4/32"]). Empty = no SG created, ALB controller manages its own. | `list(string)` | `[]` | no |
| <a name="input_availability_zones_count"></a> [availability\_zones\_count](#input\_availability\_zones\_count) | Number of availability zones to use | `number` | `3` | no |
| <a name="input_cloudwatch_logs_retention_days"></a> [cloudwatch\_logs\_retention\_days](#input\_cloudwatch\_logs\_retention\_days) | Retention period (days) for the OSMO CloudWatch log group | `number` | `30` | no |
| <a name="input_cluster_endpoint_private_access"></a> [cluster\_endpoint\_private\_access](#input\_cluster\_endpoint\_private\_access) | Enable private access to EKS API endpoint | `bool` | `true` | no |
| <a name="input_cluster_endpoint_public_access"></a> [cluster\_endpoint\_public\_access](#input\_cluster\_endpoint\_public\_access) | Enable public access to EKS API endpoint | `bool` | `true` | no |
| <a name="input_cognito_admin_email"></a> [cognito\_admin\_email](#input\_cognito\_admin\_email) | Email for the initial Cognito admin user | `string` | `""` | no |
| <a name="input_cognito_admin_temp_password"></a> [cognito\_admin\_temp\_password](#input\_cognito\_admin\_temp\_password) | Temporary password for the initial Cognito admin user (must change on first login) | `string` | `"ChangeMe123!"` | no |
| <a name="input_cognito_admin_username"></a> [cognito\_admin\_username](#input\_cognito\_admin\_username) | Username for the initial Cognito admin user | `string` | `"osmo-admin"` | no |
| <a name="input_cognito_parent_domain_placeholder_ip"></a> [cognito\_parent\_domain\_placeholder\_ip](#input\_cognito\_parent\_domain\_placeholder\_ip) | If set, Terraform creates an A record for the parent domain of the Cognito auth hostname so Cognito custom domain validation passes (e.g. 192.0.2.1). Leave empty if the parent domain already has an A record. | `string` | `"192.0.2.1"` | no |
| <a name="input_deploy_cognito"></a> [deploy\_cognito](#input\_deploy\_cognito) | Deploy AWS Cognito User Pool as the OSMO identity provider | `bool` | `false` | no |
| <a name="input_deploy_identity_center"></a> [deploy\_identity\_center](#input\_deploy\_identity\_center) | Create an IAM Identity Center OAuth 2.0 application for OSMO | `bool` | `false` | no |
| <a name="input_deploy_keycloak"></a> [deploy\_keycloak](#input\_deploy\_keycloak) | Provision ACM certificate and Terraform outputs for a self-hosted Keycloak IdP | `bool` | `true` | no |
| <a name="input_deployment_mode"></a> [deployment\_mode](#input\_deployment\_mode) | Deployment mode: full, control-plane-only, or backend-only | `string` | `"full"` | no |
| <a name="input_eks_admin_principal_arns"></a> [eks\_admin\_principal\_arns](#input\_eks\_admin\_principal\_arns) | List of IAM principal ARNs to grant EKS admin access | `list(string)` | `[]` | no |
| <a name="input_enable_cloudwatch_logging"></a> [enable\_cloudwatch\_logging](#input\_enable\_cloudwatch\_logging) | Provision the Fluent Bit IRSA role + CloudWatch log group for shipping pod logs. The Fluent Bit DaemonSet itself is deployed by 01-deploy-aws-prerequisites.sh. | `bool` | `true` | no |
| <a name="input_enable_flow_logs"></a> [enable\_flow\_logs](#input\_enable\_flow\_logs) | Enable VPC Flow Logs for network traffic analysis | `bool` | `false` | no |
| <a name="input_enable_guardduty"></a> [enable\_guardduty](#input\_enable\_guardduty) | Enable AWS GuardDuty for threat detection | `bool` | `false` | no |
| <a name="input_enable_managed_prometheus"></a> [enable\_managed\_prometheus](#input\_enable\_managed\_prometheus) | Provision an Amazon Managed Prometheus (AMP) workspace + managed scraper for cluster/GPU metrics | `bool` | `true` | no |
| <a name="input_enable_security_hub"></a> [enable\_security\_hub](#input\_enable\_security\_hub) | Enable AWS Security Hub for security posture management | `bool` | `false` | no |
| <a name="input_enable_waf"></a> [enable\_waf](#input\_enable\_waf) | Enable AWS WAF for ALB protection | `bool` | `false` | no |
| <a name="input_environment"></a> [environment](#input\_environment) | Environment name (dev, staging, prod) | `string` | `"dev"` | no |
| <a name="input_external_service_url"></a> [external\_service\_url](#input\_external\_service\_url) | External OSMO service URL (required for backend-only mode) | `string` | `null` | no |
| <a name="input_flow_logs_retention_days"></a> [flow\_logs\_retention\_days](#input\_flow\_logs\_retention\_days) | Retention period for VPC Flow Logs in CloudWatch | `number` | `30` | no |
| <a name="input_gpu_ami_id"></a> [gpu\_ami\_id](#input\_gpu\_ami\_id) | Required. AMI ID for GPU nodes — the Canonical Ubuntu EKS image for your region and Kubernetes version (see https://cloud-images.ubuntu.com/docs/aws/eks/). gpu\_ami\_type is ignored and the NVIDIA GPU Operator handles driver installation. | `string` | n/a | yes |
| <a name="input_gpu_ami_type"></a> [gpu\_ami\_type](#input\_gpu\_ami\_type) | AMI type for GPU nodes (ignored when gpu\_ami\_id is set) | `string` | `"AL2_x86_64_GPU"` | no |
| <a name="input_gpu_node_capacity_type"></a> [gpu\_node\_capacity\_type](#input\_gpu\_node\_capacity\_type) | Capacity type for GPU nodes (ON\_DEMAND or SPOT) | `string` | `"ON_DEMAND"` | no |
| <a name="input_gpu_node_desired_size"></a> [gpu\_node\_desired\_size](#input\_gpu\_node\_desired\_size) | Desired number of GPU nodes | `number` | `0` | no |
| <a name="input_gpu_node_instance_types"></a> [gpu\_node\_instance\_types](#input\_gpu\_node\_instance\_types) | Instance types for GPU node group | `list(string)` | <pre>[<br/>  "g5.2xlarge"<br/>]</pre> | no |
| <a name="input_gpu_node_max_size"></a> [gpu\_node\_max\_size](#input\_gpu\_node\_max\_size) | Maximum number of GPU nodes | `number` | `10` | no |
| <a name="input_gpu_node_min_size"></a> [gpu\_node\_min\_size](#input\_gpu\_node\_min\_size) | Minimum number of GPU nodes | `number` | `0` | no |
| <a name="input_gpu_node_volume_size"></a> [gpu\_node\_volume\_size](#input\_gpu\_node\_volume\_size) | Root EBS volume size (GiB) for GPU nodes. The 200 GiB default upstream is too small to hold the Isaac Sim (~30 GB) + Cosmos Predict2 (~25 GB) container images alongside the GPU operator driver containers — pods get evicted with 'no space left on device' during image extraction. Must be at least 200 GiB. | `number` | `500` | no |
| <a name="input_idc_admin_email"></a> [idc\_admin\_email](#input\_idc\_admin\_email) | Email of the admin user in Identity Center (legacy) | `string` | `""` | no |
| <a name="input_idc_client_secret"></a> [idc\_client\_secret](#input\_idc\_client\_secret) | OAuth2 client secret — generated in the Identity Center console after terraform apply. Leave empty on first apply. | `string` | `""` | no |
| <a name="input_idc_region"></a> [idc\_region](#input\_idc\_region) | AWS region where Identity Center is enabled (defaults to aws\_region) | `string` | `""` | no |
| <a name="input_keycloak_admin_username"></a> [keycloak\_admin\_username](#input\_keycloak\_admin\_username) | Username of the OSMO admin user inside Keycloak (used as x-osmo-user identity) | `string` | `"admin"` | no |
| <a name="input_keycloak_realm"></a> [keycloak\_realm](#input\_keycloak\_realm) | Keycloak realm name for OSMO | `string` | `"osmo"` | no |
| <a name="input_kms_enable_key_rotation"></a> [kms\_enable\_key\_rotation](#input\_kms\_enable\_key\_rotation) | Enable automatic key rotation for KMS keys | `bool` | `true` | no |
| <a name="input_kms_key_deletion_window_days"></a> [kms\_key\_deletion\_window\_days](#input\_kms\_key\_deletion\_window\_days) | Waiting period before KMS key deletion | `number` | `7` | no |
| <a name="input_kubernetes_version"></a> [kubernetes\_version](#input\_kubernetes\_version) | Kubernetes version for EKS cluster | `string` | `"1.35"` | no |
| <a name="input_owner"></a> [owner](#input\_owner) | Owner tag for resources | `string` | `"osmo-team"` | no |
| <a name="input_project_name"></a> [project\_name](#input\_project\_name) | Project name for resource tagging | `string` | `"osmo"` | no |
| <a name="input_rds_allocated_storage"></a> [rds\_allocated\_storage](#input\_rds\_allocated\_storage) | Allocated storage in GB | `number` | `20` | no |
| <a name="input_rds_backup_retention_period"></a> [rds\_backup\_retention\_period](#input\_rds\_backup\_retention\_period) | Backup retention period in days | `number` | `7` | no |
| <a name="input_rds_db_name"></a> [rds\_db\_name](#input\_rds\_db\_name) | Name of the database to create | `string` | `"osmo"` | no |
| <a name="input_rds_deletion_protection"></a> [rds\_deletion\_protection](#input\_rds\_deletion\_protection) | Enable deletion protection | `bool` | `false` | no |
| <a name="input_rds_engine_version"></a> [rds\_engine\_version](#input\_rds\_engine\_version) | PostgreSQL engine version | `string` | `"15.17"` | no |
| <a name="input_rds_instance_class"></a> [rds\_instance\_class](#input\_rds\_instance\_class) | RDS instance class | `string` | `"db.t3.medium"` | no |
| <a name="input_rds_max_allocated_storage"></a> [rds\_max\_allocated\_storage](#input\_rds\_max\_allocated\_storage) | Maximum allocated storage in GB (autoscaling) | `number` | `100` | no |
| <a name="input_rds_multi_az"></a> [rds\_multi\_az](#input\_rds\_multi\_az) | Enable Multi-AZ deployment | `bool` | `false` | no |
| <a name="input_rds_username"></a> [rds\_username](#input\_rds\_username) | Master username for RDS | `string` | `"osmo_admin"` | no |
| <a name="input_redis_automatic_failover_enabled"></a> [redis\_automatic\_failover\_enabled](#input\_redis\_automatic\_failover\_enabled) | Enable automatic failover (requires num\_cache\_clusters >= 2) | `bool` | `false` | no |
| <a name="input_redis_engine_version"></a> [redis\_engine\_version](#input\_redis\_engine\_version) | Redis engine version | `string` | `"7.1"` | no |
| <a name="input_redis_multi_az_enabled"></a> [redis\_multi\_az\_enabled](#input\_redis\_multi\_az\_enabled) | Enable Multi-AZ for Redis | `bool` | `false` | no |
| <a name="input_redis_node_type"></a> [redis\_node\_type](#input\_redis\_node\_type) | ElastiCache node type | `string` | `"cache.t3.medium"` | no |
| <a name="input_redis_num_cache_clusters"></a> [redis\_num\_cache\_clusters](#input\_redis\_num\_cache\_clusters) | Number of cache clusters (nodes) in the replication group | `number` | `1` | no |
| <a name="input_redis_snapshot_retention_limit"></a> [redis\_snapshot\_retention\_limit](#input\_redis\_snapshot\_retention\_limit) | Number of days to retain automatic snapshots | `number` | `1` | no |
| <a name="input_restrict_rds_to_eks_only"></a> [restrict\_rds\_to\_eks\_only](#input\_restrict\_rds\_to\_eks\_only) | Restrict RDS access to EKS security group only (no VPC CIDR access) | `bool` | `false` | no |
| <a name="input_restrict_redis_to_eks_only"></a> [restrict\_redis\_to\_eks\_only](#input\_restrict\_redis\_to\_eks\_only) | Restrict Redis access to EKS security group only (no VPC CIDR access) | `bool` | `false` | no |
| <a name="input_s3_allowed_origins"></a> [s3\_allowed\_origins](#input\_s3\_allowed\_origins) | Allowed origins for S3 CORS (empty = allow all, production should specify domains) | `list(string)` | `[]` | no |
| <a name="input_s3_force_destroy"></a> [s3\_force\_destroy](#input\_s3\_force\_destroy) | Allow S3 buckets to be destroyed with objects inside | `bool` | `false` | no |
| <a name="input_s3_versioning_enabled"></a> [s3\_versioning\_enabled](#input\_s3\_versioning\_enabled) | Enable versioning for S3 buckets | `bool` | `true` | no |
| <a name="input_secrets_recovery_window_days"></a> [secrets\_recovery\_window\_days](#input\_secrets\_recovery\_window\_days) | Number of days AWS Secrets Manager waits before deletion | `number` | `7` | no |
| <a name="input_should_deploy_gpu_nodes"></a> [should\_deploy\_gpu\_nodes](#input\_should\_deploy\_gpu\_nodes) | Deploy GPU node group | `bool` | `true` | no |
| <a name="input_should_deploy_postgresql"></a> [should\_deploy\_postgresql](#input\_should\_deploy\_postgresql) | Deploy RDS PostgreSQL instance | `bool` | `true` | no |
| <a name="input_should_deploy_redis"></a> [should\_deploy\_redis](#input\_should\_deploy\_redis) | Deploy ElastiCache Redis cluster | `bool` | `true` | no |
| <a name="input_single_nat_gateway"></a> [single\_nat\_gateway](#input\_single\_nat\_gateway) | Use a single NAT gateway (cost savings for non-prod) | `bool` | `true` | no |
| <a name="input_system_node_desired_size"></a> [system\_node\_desired\_size](#input\_system\_node\_desired\_size) | Desired number of system nodes | `number` | `3` | no |
| <a name="input_system_node_instance_types"></a> [system\_node\_instance\_types](#input\_system\_node\_instance\_types) | Instance types for system node group | `list(string)` | <pre>[<br/>  "m6i.xlarge"<br/>]</pre> | no |
| <a name="input_system_node_max_size"></a> [system\_node\_max\_size](#input\_system\_node\_max\_size) | Maximum number of system nodes | `number` | `6` | no |
| <a name="input_system_node_min_size"></a> [system\_node\_min\_size](#input\_system\_node\_min\_size) | Minimum number of system nodes | `number` | `2` | no |
| <a name="input_vpc_cidr"></a> [vpc\_cidr](#input\_vpc\_cidr) | CIDR block for the VPC | `string` | `"10.0.0.0/16"` | no |
| <a name="input_waf_allowed_ip_cidrs"></a> [waf\_allowed\_ip\_cidrs](#input\_waf\_allowed\_ip\_cidrs) | List of CIDR blocks allowed to access the ALB (empty = allow all) | `list(string)` | `[]` | no |
| <a name="input_waf_block_mode"></a> [waf\_block\_mode](#input\_waf\_block\_mode) | WAF action mode: COUNT (monitor) or BLOCK (enforce) | `string` | `"BLOCK"` | no |
| <a name="input_waf_rate_limit"></a> [waf\_rate\_limit](#input\_waf\_rate\_limit) | Rate limit for requests per 5-minute period per IP | `number` | `2000` | no |

## Outputs

## Outputs

| Name | Description |
|------|-------------|
| <a name="output_acm_auth_certificate_arn"></a> [acm\_auth\_certificate\_arn](#output\_acm\_auth\_certificate\_arn) | ACM certificate ARN for OSMO auth domain |
| <a name="output_acm_certificate_arn"></a> [acm\_certificate\_arn](#output\_acm\_certificate\_arn) | ACM certificate ARN for OSMO service domain |
| <a name="output_alb_security_group_id"></a> [alb\_security\_group\_id](#output\_alb\_security\_group\_id) | ALB security group ID (null when alb\_allowed\_cidrs is empty) |
| <a name="output_amp_workspace_id"></a> [amp\_workspace\_id](#output\_amp\_workspace\_id) | Amazon Managed Prometheus workspace ID |
| <a name="output_amp_workspace_prometheus_endpoint"></a> [amp\_workspace\_prometheus\_endpoint](#output\_amp\_workspace\_prometheus\_endpoint) | Amazon Managed Prometheus remote-write/query endpoint |
| <a name="output_aws_lb_controller_role_arn"></a> [aws\_lb\_controller\_role\_arn](#output\_aws\_lb\_controller\_role\_arn) | IAM role ARN for AWS Load Balancer Controller |
| <a name="output_aws_region"></a> [aws\_region](#output\_aws\_region) | AWS region |
| <a name="output_cloudwatch_log_group_name"></a> [cloudwatch\_log\_group\_name](#output\_cloudwatch\_log\_group\_name) | CloudWatch log group for OSMO container logs |
| <a name="output_cluster_autoscaler_role_arn"></a> [cluster\_autoscaler\_role\_arn](#output\_cluster\_autoscaler\_role\_arn) | IAM role ARN for Cluster Autoscaler |
| <a name="output_cluster_ca_certificate"></a> [cluster\_ca\_certificate](#output\_cluster\_ca\_certificate) | Base64 encoded cluster CA certificate |
| <a name="output_cluster_endpoint"></a> [cluster\_endpoint](#output\_cluster\_endpoint) | EKS cluster API endpoint |
| <a name="output_cluster_name"></a> [cluster\_name](#output\_cluster\_name) | EKS cluster name |
| <a name="output_cluster_oidc_issuer_url"></a> [cluster\_oidc\_issuer\_url](#output\_cluster\_oidc\_issuer\_url) | OIDC issuer URL for the EKS cluster |
| <a name="output_cluster_oidc_provider_arn"></a> [cluster\_oidc\_provider\_arn](#output\_cluster\_oidc\_provider\_arn) | OIDC provider ARN for IRSA |
| <a name="output_cognito_authorize_endpoint"></a> [cognito\_authorize\_endpoint](#output\_cognito\_authorize\_endpoint) | Cognito OAuth2 authorize endpoint |
| <a name="output_cognito_browser_client_id"></a> [cognito\_browser\_client\_id](#output\_cognito\_browser\_client\_id) | Cognito browser app client ID |
| <a name="output_cognito_browser_client_secret"></a> [cognito\_browser\_client\_secret](#output\_cognito\_browser\_client\_secret) | Cognito browser app client secret |
| <a name="output_cognito_cli_client_id"></a> [cognito\_cli\_client\_id](#output\_cognito\_cli\_client\_id) | Cognito CLI app client ID |
| <a name="output_cognito_domain"></a> [cognito\_domain](#output\_cognito\_domain) | Cognito custom domain FQDN |
| <a name="output_cognito_issuer_url"></a> [cognito\_issuer\_url](#output\_cognito\_issuer\_url) | Cognito OIDC issuer URL |
| <a name="output_cognito_jwks_uri"></a> [cognito\_jwks\_uri](#output\_cognito\_jwks\_uri) | Cognito JWKS URI for JWT validation |
| <a name="output_cognito_logout_endpoint"></a> [cognito\_logout\_endpoint](#output\_cognito\_logout\_endpoint) | Cognito logout endpoint |
| <a name="output_cognito_token_endpoint"></a> [cognito\_token\_endpoint](#output\_cognito\_token\_endpoint) | Cognito OAuth2 token endpoint |
| <a name="output_cognito_user_pool_id"></a> [cognito\_user\_pool\_id](#output\_cognito\_user\_pool\_id) | Cognito User Pool ID |
| <a name="output_configure_kubectl"></a> [configure\_kubectl](#output\_configure\_kubectl) | AWS CLI command to configure kubectl |
| <a name="output_deployment_mode"></a> [deployment\_mode](#output\_deployment\_mode) | Deployment mode |
| <a name="output_ebs_csi_driver_role_arn"></a> [ebs\_csi\_driver\_role\_arn](#output\_ebs\_csi\_driver\_role\_arn) | IAM role ARN for EBS CSI driver |
| <a name="output_environment"></a> [environment](#output\_environment) | Environment name |
| <a name="output_external_dns_role_arn"></a> [external\_dns\_role\_arn](#output\_external\_dns\_role\_arn) | IAM role ARN for external-dns |
| <a name="output_external_secrets_role_arn"></a> [external\_secrets\_role\_arn](#output\_external\_secrets\_role\_arn) | IAM role ARN for External Secrets Operator |
| <a name="output_external_service_url"></a> [external\_service\_url](#output\_external\_service\_url) | External OSMO service URL (for backend-only mode) |
| <a name="output_fluentbit_role_arn"></a> [fluentbit\_role\_arn](#output\_fluentbit\_role\_arn) | IAM role ARN for the Fluent Bit log shipper (IRSA) |
| <a name="output_idc_admin_email"></a> [idc\_admin\_email](#output\_idc\_admin\_email) | Identity Center admin user email |
| <a name="output_idc_application_arn"></a> [idc\_application\_arn](#output\_idc\_application\_arn) | Identity Center application ARN |
| <a name="output_idc_authorize_endpoint"></a> [idc\_authorize\_endpoint](#output\_idc\_authorize\_endpoint) | Identity Center OAuth2 authorize endpoint |
| <a name="output_idc_client_id"></a> [idc\_client\_id](#output\_idc\_client\_id) | Identity Center OAuth2 client ID (same as application ARN) |
| <a name="output_idc_client_secret"></a> [idc\_client\_secret](#output\_idc\_client\_secret) | Identity Center OAuth2 client secret (from console) |
| <a name="output_idc_instance_id"></a> [idc\_instance\_id](#output\_idc\_instance\_id) | Identity Center instance ID (e.g. ssoins-abc123def456) |
| <a name="output_idc_issuer_url"></a> [idc\_issuer\_url](#output\_idc\_issuer\_url) | Identity Center OIDC issuer URL |
| <a name="output_idc_jwks_uri"></a> [idc\_jwks\_uri](#output\_idc\_jwks\_uri) | Identity Center JWKS URI for JWT validation |
| <a name="output_idc_token_endpoint"></a> [idc\_token\_endpoint](#output\_idc\_token\_endpoint) | Identity Center OAuth2 token endpoint |
| <a name="output_keycloak_admin_username"></a> [keycloak\_admin\_username](#output\_keycloak\_admin\_username) | Keycloak admin username for OSMO bootstrap |
| <a name="output_keycloak_authorize_endpoint"></a> [keycloak\_authorize\_endpoint](#output\_keycloak\_authorize\_endpoint) | Keycloak OAuth2 authorize endpoint |
| <a name="output_keycloak_browser_client_id"></a> [keycloak\_browser\_client\_id](#output\_keycloak\_browser\_client\_id) | Keycloak browser-flow client ID |
| <a name="output_keycloak_device_client_id"></a> [keycloak\_device\_client\_id](#output\_keycloak\_device\_client\_id) | Keycloak device-flow (CLI) client ID |
| <a name="output_keycloak_device_endpoint"></a> [keycloak\_device\_endpoint](#output\_keycloak\_device\_endpoint) | Keycloak OAuth2 device authorization endpoint |
| <a name="output_keycloak_issuer_url"></a> [keycloak\_issuer\_url](#output\_keycloak\_issuer\_url) | Keycloak OIDC issuer URL |
| <a name="output_keycloak_jwks_uri"></a> [keycloak\_jwks\_uri](#output\_keycloak\_jwks\_uri) | Keycloak JWKS URI for JWT validation |
| <a name="output_keycloak_logout_endpoint"></a> [keycloak\_logout\_endpoint](#output\_keycloak\_logout\_endpoint) | Keycloak logout endpoint |
| <a name="output_keycloak_token_endpoint"></a> [keycloak\_token\_endpoint](#output\_keycloak\_token\_endpoint) | Keycloak OAuth2 token endpoint |
| <a name="output_kms_key_arn"></a> [kms\_key\_arn](#output\_kms\_key\_arn) | KMS key ARN for encryption |
| <a name="output_kms_key_id"></a> [kms\_key\_id](#output\_kms\_key\_id) | KMS key ID |
| <a name="output_osmo_auth_hostname"></a> [osmo\_auth\_hostname](#output\_osmo\_auth\_hostname) | Hostname for OSMO auth/Keycloak Ingress |
| <a name="output_osmo_backend_role_arn"></a> [osmo\_backend\_role\_arn](#output\_osmo\_backend\_role\_arn) | IAM role ARN for OSMO backend operator |
| <a name="output_osmo_backend_service_account"></a> [osmo\_backend\_service\_account](#output\_osmo\_backend\_service\_account) | Service account name for OSMO backend operator |
| <a name="output_osmo_hostname"></a> [osmo\_hostname](#output\_osmo\_hostname) | Hostname for OSMO service Ingress |
| <a name="output_osmo_namespace"></a> [osmo\_namespace](#output\_osmo\_namespace) | Kubernetes namespace for OSMO |
| <a name="output_osmo_s3_credentials_secret_arn"></a> [osmo\_s3\_credentials\_secret\_arn](#output\_osmo\_s3\_credentials\_secret\_arn) | ARN of the Secrets Manager secret containing OSMO S3 IAM user credentials |
| <a name="output_osmo_service_account"></a> [osmo\_service\_account](#output\_osmo\_service\_account) | Service account name for OSMO service |
| <a name="output_osmo_service_role_arn"></a> [osmo\_service\_role\_arn](#output\_osmo\_service\_role\_arn) | IAM role ARN for OSMO service |
| <a name="output_osmo_workflow_role_arn"></a> [osmo\_workflow\_role\_arn](#output\_osmo\_workflow\_role\_arn) | IAM role ARN for OSMO workflow task pods (IRSA dataset S3 access) |
| <a name="output_osmo_workflow_service_account"></a> [osmo\_workflow\_service\_account](#output\_osmo\_workflow\_service\_account) | Service account name used by OSMO workflow task pods (IRSA) |
| <a name="output_osmo_workflows_namespace"></a> [osmo\_workflows\_namespace](#output\_osmo\_workflows\_namespace) | Kubernetes namespace where OSMO workflow task pods run |
| <a name="output_private_subnet_ids"></a> [private\_subnet\_ids](#output\_private\_subnet\_ids) | List of private subnet IDs |
| <a name="output_public_subnet_ids"></a> [public\_subnet\_ids](#output\_public\_subnet\_ids) | List of public subnet IDs |
| <a name="output_rds_database_name"></a> [rds\_database\_name](#output\_rds\_database\_name) | RDS database name |
| <a name="output_rds_endpoint"></a> [rds\_endpoint](#output\_rds\_endpoint) | RDS instance endpoint |
| <a name="output_rds_password_secret_arn"></a> [rds\_password\_secret\_arn](#output\_rds\_password\_secret\_arn) | ARN of the Secrets Manager secret containing RDS password |
| <a name="output_rds_port"></a> [rds\_port](#output\_rds\_port) | RDS instance port |
| <a name="output_rds_username"></a> [rds\_username](#output\_rds\_username) | RDS master username |
| <a name="output_redis_auth_token_secret_arn"></a> [redis\_auth\_token\_secret\_arn](#output\_redis\_auth\_token\_secret\_arn) | ARN of the Secrets Manager secret containing Redis auth token |
| <a name="output_redis_endpoint"></a> [redis\_endpoint](#output\_redis\_endpoint) | Redis primary endpoint |
| <a name="output_redis_port"></a> [redis\_port](#output\_redis\_port) | Redis port |
| <a name="output_resource_suffix"></a> [resource\_suffix](#output\_resource\_suffix) | Resource suffix used for naming (avoids collisions between deployments) |
| <a name="output_route53_zone_id"></a> [route53\_zone\_id](#output\_route53\_zone\_id) | Route53 hosted zone ID |
| <a name="output_s3_datasets_bucket_arn"></a> [s3\_datasets\_bucket\_arn](#output\_s3\_datasets\_bucket\_arn) | S3 bucket ARN for datasets |
| <a name="output_s3_datasets_bucket_name"></a> [s3\_datasets\_bucket\_name](#output\_s3\_datasets\_bucket\_name) | S3 bucket name for datasets |
| <a name="output_s3_region"></a> [s3\_region](#output\_s3\_region) | S3 region |
| <a name="output_s3_workflows_bucket_arn"></a> [s3\_workflows\_bucket\_arn](#output\_s3\_workflows\_bucket\_arn) | S3 bucket ARN for workflows |
| <a name="output_s3_workflows_bucket_name"></a> [s3\_workflows\_bucket\_name](#output\_s3\_workflows\_bucket\_name) | S3 bucket name for workflows |
| <a name="output_secrets_manager_arn"></a> [secrets\_manager\_arn](#output\_secrets\_manager\_arn) | Secrets Manager secret ARN for OSMO secrets |
| <a name="output_vpc_id"></a> [vpc\_id](#output\_vpc\_id) | VPC ID |
| <a name="output_waf_web_acl_arn"></a> [waf\_web\_acl\_arn](#output\_waf\_web\_acl\_arn) | WAF Web ACL ARN (for ALB association) |

<!-- END_TF_DOCS -->