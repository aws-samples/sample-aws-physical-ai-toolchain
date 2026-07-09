<!-- BEGIN_TF_DOCS -->


## Requirements

## Requirements

| Name | Version |
|------|---------|
| <a name="requirement_terraform"></a> [terraform](#requirement\_terraform) | >= 1.5.0 |
| <a name="requirement_aws"></a> [aws](#requirement\_aws) | >= 5.0 |

## Providers

## Providers

| Name | Version |
|------|---------|
| <a name="provider_aws"></a> [aws](#provider\_aws) | >= 5.0 |

## Modules

## Modules

| Name | Source | Version |
|------|--------|---------|
| <a name="module_aws_lb_controller_irsa"></a> [aws\_lb\_controller\_irsa](#module\_aws\_lb\_controller\_irsa) | terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks | ~> 5.0 |
| <a name="module_cluster_autoscaler_irsa"></a> [cluster\_autoscaler\_irsa](#module\_cluster\_autoscaler\_irsa) | terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks | ~> 5.0 |
| <a name="module_ebs_csi_irsa"></a> [ebs\_csi\_irsa](#module\_ebs\_csi\_irsa) | terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks | ~> 5.0 |
| <a name="module_eks"></a> [eks](#module\_eks) | terraform-aws-modules/eks/aws | ~> 20.0 |
| <a name="module_external_dns_irsa"></a> [external\_dns\_irsa](#module\_external\_dns\_irsa) | terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks | ~> 5.0 |
| <a name="module_external_secrets_irsa"></a> [external\_secrets\_irsa](#module\_external\_secrets\_irsa) | terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks | ~> 5.0 |
| <a name="module_osmo_backend_irsa"></a> [osmo\_backend\_irsa](#module\_osmo\_backend\_irsa) | terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks | ~> 5.0 |
| <a name="module_osmo_service_irsa"></a> [osmo\_service\_irsa](#module\_osmo\_service\_irsa) | terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks | ~> 5.0 |
| <a name="module_osmo_workflow_irsa"></a> [osmo\_workflow\_irsa](#module\_osmo\_workflow\_irsa) | terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks | ~> 5.0 |
| <a name="module_vpc_cni_irsa"></a> [vpc\_cni\_irsa](#module\_vpc\_cni\_irsa) | terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks | ~> 5.0 |

## Resources

## Resources

| Name | Type |
|------|------|
| [aws_iam_policy.osmo_backend_ecr](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_policy) | resource |
| [aws_iam_policy.osmo_backend_kms](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_policy) | resource |
| [aws_iam_policy.osmo_backend_s3](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_policy) | resource |
| [aws_iam_policy.osmo_backend_secrets](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_policy) | resource |
| [aws_iam_policy.osmo_service_kms](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_policy) | resource |
| [aws_iam_policy.osmo_service_s3](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_policy) | resource |
| [aws_iam_policy.osmo_service_secrets](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_policy) | resource |
| [aws_iam_policy.osmo_workflow_s3](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_policy) | resource |
| [aws_iam_policy.vpc_cni_extra](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/iam_policy) | resource |

## Inputs

## Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|:--------:|
| <a name="input_aws_lb_controller_sa"></a> [aws\_lb\_controller\_sa](#input\_aws\_lb\_controller\_sa) | Service account name for AWS Load Balancer Controller | `string` | n/a | yes |
| <a name="input_cluster_autoscaler_sa"></a> [cluster\_autoscaler\_sa](#input\_cluster\_autoscaler\_sa) | Service account name for Cluster Autoscaler | `string` | n/a | yes |
| <a name="input_cluster_endpoint_private_access"></a> [cluster\_endpoint\_private\_access](#input\_cluster\_endpoint\_private\_access) | Enable private access to cluster endpoint | `bool` | n/a | yes |
| <a name="input_cluster_endpoint_public_access"></a> [cluster\_endpoint\_public\_access](#input\_cluster\_endpoint\_public\_access) | Enable public access to cluster endpoint | `bool` | n/a | yes |
| <a name="input_cluster_name"></a> [cluster\_name](#input\_cluster\_name) | EKS cluster name | `string` | n/a | yes |
| <a name="input_common_tags"></a> [common\_tags](#input\_common\_tags) | Common tags for all resources | `map(string)` | n/a | yes |
| <a name="input_deploy_gpu_nodes"></a> [deploy\_gpu\_nodes](#input\_deploy\_gpu\_nodes) | Deploy GPU node group | `bool` | n/a | yes |
| <a name="input_eks_admin_principal_arns"></a> [eks\_admin\_principal\_arns](#input\_eks\_admin\_principal\_arns) | IAM principal ARNs for EKS admin access | `list(string)` | n/a | yes |
| <a name="input_external_secrets_sa"></a> [external\_secrets\_sa](#input\_external\_secrets\_sa) | Service account name for External Secrets Operator | `string` | n/a | yes |
| <a name="input_gpu_ami_type"></a> [gpu\_ami\_type](#input\_gpu\_ami\_type) | AMI type for GPU nodes (ignored when gpu\_ami\_id is set) | `string` | n/a | yes |
| <a name="input_gpu_node_capacity_type"></a> [gpu\_node\_capacity\_type](#input\_gpu\_node\_capacity\_type) | Capacity type for GPU nodes | `string` | n/a | yes |
| <a name="input_gpu_node_desired_size"></a> [gpu\_node\_desired\_size](#input\_gpu\_node\_desired\_size) | Desired number of GPU nodes | `number` | n/a | yes |
| <a name="input_gpu_node_instance_types"></a> [gpu\_node\_instance\_types](#input\_gpu\_node\_instance\_types) | Instance types for GPU node group | `list(string)` | n/a | yes |
| <a name="input_gpu_node_max_size"></a> [gpu\_node\_max\_size](#input\_gpu\_node\_max\_size) | Maximum number of GPU nodes | `number` | n/a | yes |
| <a name="input_gpu_node_min_size"></a> [gpu\_node\_min\_size](#input\_gpu\_node\_min\_size) | Minimum number of GPU nodes | `number` | n/a | yes |
| <a name="input_gpu_taints"></a> [gpu\_taints](#input\_gpu\_taints) | Taints for GPU node group | <pre>list(object({<br/>    key    = string<br/>    value  = string<br/>    effect = string<br/>  }))</pre> | n/a | yes |
| <a name="input_kms_key_arn"></a> [kms\_key\_arn](#input\_kms\_key\_arn) | KMS key ARN for encryption | `string` | n/a | yes |
| <a name="input_kubernetes_version"></a> [kubernetes\_version](#input\_kubernetes\_version) | Kubernetes version | `string` | n/a | yes |
| <a name="input_name_prefix"></a> [name\_prefix](#input\_name\_prefix) | Prefix for resource names | `string` | n/a | yes |
| <a name="input_osmo_backend_listener_sa"></a> [osmo\_backend\_listener\_sa](#input\_osmo\_backend\_listener\_sa) | ServiceAccount name created by the backend-operator chart for the listener (release-name prefixed) | `string` | n/a | yes |
| <a name="input_osmo_backend_worker_sa"></a> [osmo\_backend\_worker\_sa](#input\_osmo\_backend\_worker\_sa) | ServiceAccount name created by the backend-operator chart for the worker (release-name prefixed) | `string` | n/a | yes |
| <a name="input_osmo_namespace"></a> [osmo\_namespace](#input\_osmo\_namespace) | Kubernetes namespace for OSMO | `string` | n/a | yes |
| <a name="input_osmo_operator_namespace"></a> [osmo\_operator\_namespace](#input\_osmo\_operator\_namespace) | Kubernetes namespace where the OSMO backend operator runs | `string` | n/a | yes |
| <a name="input_osmo_service_account"></a> [osmo\_service\_account](#input\_osmo\_service\_account) | Service account name for OSMO service | `string` | n/a | yes |
| <a name="input_osmo_workflow_sa"></a> [osmo\_workflow\_sa](#input\_osmo\_workflow\_sa) | Service account name used by OSMO workflow task pods (IRSA for dataset S3 access) | `string` | n/a | yes |
| <a name="input_osmo_workflows_namespace"></a> [osmo\_workflows\_namespace](#input\_osmo\_workflows\_namespace) | Kubernetes namespace where OSMO workflow task pods run | `string` | n/a | yes |
| <a name="input_private_subnets"></a> [private\_subnets](#input\_private\_subnets) | Private subnet IDs | `list(string)` | n/a | yes |
| <a name="input_route53_zone_id"></a> [route53\_zone\_id](#input\_route53\_zone\_id) | Route53 hosted zone ID for external-dns | `string` | n/a | yes |
| <a name="input_s3_datasets_bucket_arn"></a> [s3\_datasets\_bucket\_arn](#input\_s3\_datasets\_bucket\_arn) | S3 datasets bucket ARN | `string` | n/a | yes |
| <a name="input_s3_workflows_bucket_arn"></a> [s3\_workflows\_bucket\_arn](#input\_s3\_workflows\_bucket\_arn) | S3 workflows bucket ARN | `string` | n/a | yes |
| <a name="input_secrets_manager_arn"></a> [secrets\_manager\_arn](#input\_secrets\_manager\_arn) | Secrets Manager ARN for OSMO secrets | `string` | n/a | yes |
| <a name="input_system_node_desired_size"></a> [system\_node\_desired\_size](#input\_system\_node\_desired\_size) | Desired number of system nodes | `number` | n/a | yes |
| <a name="input_system_node_instance_types"></a> [system\_node\_instance\_types](#input\_system\_node\_instance\_types) | Instance types for system node group | `list(string)` | n/a | yes |
| <a name="input_system_node_max_size"></a> [system\_node\_max\_size](#input\_system\_node\_max\_size) | Maximum number of system nodes | `number` | n/a | yes |
| <a name="input_system_node_min_size"></a> [system\_node\_min\_size](#input\_system\_node\_min\_size) | Minimum number of system nodes | `number` | n/a | yes |
| <a name="input_vpc_id"></a> [vpc\_id](#input\_vpc\_id) | VPC ID | `string` | n/a | yes |
| <a name="input_gpu_ami_id"></a> [gpu\_ami\_id](#input\_gpu\_ami\_id) | Custom AMI ID for GPU nodes (e.g. Canonical Ubuntu EKS image). When set, gpu\_ami\_type is ignored and bootstrap user data is auto-generated. | `string` | `""` | no |
| <a name="input_gpu_node_volume_size"></a> [gpu\_node\_volume\_size](#input\_gpu\_node\_volume\_size) | Root EBS volume size (GiB) for GPU nodes. Must be large enough to hold container images (Isaac Sim ~30 GB, Cosmos Predict2 ~25 GB), GPU operator drivers, and the kubelet's image-gc + eviction reserves. | `number` | `500` | no |

## Outputs

## Outputs

| Name | Description |
|------|-------------|
| <a name="output_aws_lb_controller_role_arn"></a> [aws\_lb\_controller\_role\_arn](#output\_aws\_lb\_controller\_role\_arn) | AWS Load Balancer Controller IRSA role ARN |
| <a name="output_cluster_arn"></a> [cluster\_arn](#output\_cluster\_arn) | EKS cluster ARN |
| <a name="output_cluster_autoscaler_role_arn"></a> [cluster\_autoscaler\_role\_arn](#output\_cluster\_autoscaler\_role\_arn) | Cluster Autoscaler IRSA role ARN |
| <a name="output_cluster_ca_certificate"></a> [cluster\_ca\_certificate](#output\_cluster\_ca\_certificate) | Base64 encoded cluster CA certificate |
| <a name="output_cluster_endpoint"></a> [cluster\_endpoint](#output\_cluster\_endpoint) | EKS cluster API endpoint |
| <a name="output_cluster_name"></a> [cluster\_name](#output\_cluster\_name) | EKS cluster name |
| <a name="output_cluster_oidc_issuer_url"></a> [cluster\_oidc\_issuer\_url](#output\_cluster\_oidc\_issuer\_url) | OIDC issuer URL |
| <a name="output_cluster_primary_security_group_id"></a> [cluster\_primary\_security\_group\_id](#output\_cluster\_primary\_security\_group\_id) | Cluster primary security group ID |
| <a name="output_cluster_security_group_id"></a> [cluster\_security\_group\_id](#output\_cluster\_security\_group\_id) | Cluster security group ID |
| <a name="output_ebs_csi_driver_role_arn"></a> [ebs\_csi\_driver\_role\_arn](#output\_ebs\_csi\_driver\_role\_arn) | EBS CSI driver IRSA role ARN |
| <a name="output_eks_managed_node_groups"></a> [eks\_managed\_node\_groups](#output\_eks\_managed\_node\_groups) | Map of EKS managed node groups |
| <a name="output_external_dns_role_arn"></a> [external\_dns\_role\_arn](#output\_external\_dns\_role\_arn) | external-dns IRSA role ARN |
| <a name="output_external_secrets_role_arn"></a> [external\_secrets\_role\_arn](#output\_external\_secrets\_role\_arn) | External Secrets Operator IRSA role ARN |
| <a name="output_gpu_node_group_arn"></a> [gpu\_node\_group\_arn](#output\_gpu\_node\_group\_arn) | GPU node group ARN |
| <a name="output_node_security_group_id"></a> [node\_security\_group\_id](#output\_node\_security\_group\_id) | Node security group ID |
| <a name="output_oidc_provider_arn"></a> [oidc\_provider\_arn](#output\_oidc\_provider\_arn) | OIDC provider ARN for IRSA |
| <a name="output_osmo_backend_role_arn"></a> [osmo\_backend\_role\_arn](#output\_osmo\_backend\_role\_arn) | OSMO backend operator IRSA role ARN |
| <a name="output_osmo_service_role_arn"></a> [osmo\_service\_role\_arn](#output\_osmo\_service\_role\_arn) | OSMO service IRSA role ARN |
| <a name="output_osmo_workflow_role_arn"></a> [osmo\_workflow\_role\_arn](#output\_osmo\_workflow\_role\_arn) | IAM role ARN for OSMO workflow task pods (IRSA dataset S3 access) |
| <a name="output_system_node_group_arn"></a> [system\_node\_group\_arn](#output\_system\_node\_group\_arn) | System node group ARN |
| <a name="output_vpc_cni_role_arn"></a> [vpc\_cni\_role\_arn](#output\_vpc\_cni\_role\_arn) | VPC CNI IRSA role ARN |

<!-- END_TF_DOCS -->