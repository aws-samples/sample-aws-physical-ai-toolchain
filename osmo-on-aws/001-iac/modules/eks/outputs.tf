# SPDX-License-Identifier: Apache-2.0

#------------------------------------------------------------------------------
# Cluster Outputs
#------------------------------------------------------------------------------

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "EKS cluster API endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_arn" {
  description = "EKS cluster ARN"
  value       = module.eks.cluster_arn
}

output "cluster_ca_certificate" {
  description = "Base64 encoded cluster CA certificate"
  value       = module.eks.cluster_certificate_authority_data
}

output "cluster_oidc_issuer_url" {
  description = "OIDC issuer URL"
  value       = module.eks.cluster_oidc_issuer_url
}

output "oidc_provider_arn" {
  description = "OIDC provider ARN for IRSA"
  value       = module.eks.oidc_provider_arn
}

output "cluster_primary_security_group_id" {
  description = "Cluster primary security group ID"
  value       = module.eks.cluster_primary_security_group_id
}

output "cluster_security_group_id" {
  description = "Cluster security group ID"
  value       = module.eks.cluster_security_group_id
}

output "node_security_group_id" {
  description = "Node security group ID"
  value       = module.eks.node_security_group_id
}

#------------------------------------------------------------------------------
# Node Group Outputs
#------------------------------------------------------------------------------

output "eks_managed_node_groups" {
  description = "Map of EKS managed node groups"
  value       = module.eks.eks_managed_node_groups
}

output "system_node_group_arn" {
  description = "System node group ARN"
  value       = try(module.eks.eks_managed_node_groups["system"].node_group_arn, null)
}

output "gpu_node_group_arn" {
  description = "GPU node group ARN"
  value       = try(module.eks.eks_managed_node_groups["gpu"].node_group_arn, null)
}

#------------------------------------------------------------------------------
# IRSA Role ARNs
#------------------------------------------------------------------------------

output "osmo_service_role_arn" {
  description = "OSMO service IRSA role ARN"
  value       = module.osmo_service_irsa.iam_role_arn
}

output "osmo_workflow_role_arn" {
  description = "IAM role ARN for OSMO workflow task pods (IRSA dataset S3 access)"
  value       = module.osmo_workflow_irsa.iam_role_arn
}

output "osmo_backend_role_arn" {
  description = "OSMO backend operator IRSA role ARN"
  value       = module.osmo_backend_irsa.iam_role_arn
}

output "aws_lb_controller_role_arn" {
  description = "AWS Load Balancer Controller IRSA role ARN"
  value       = module.aws_lb_controller_irsa.iam_role_arn
}

output "ebs_csi_driver_role_arn" {
  description = "EBS CSI driver IRSA role ARN"
  value       = module.ebs_csi_irsa.iam_role_arn
}

output "cluster_autoscaler_role_arn" {
  description = "Cluster Autoscaler IRSA role ARN"
  value       = module.cluster_autoscaler_irsa.iam_role_arn
}

output "external_secrets_role_arn" {
  description = "External Secrets Operator IRSA role ARN"
  value       = module.external_secrets_irsa.iam_role_arn
}

output "external_dns_role_arn" {
  description = "external-dns IRSA role ARN"
  value       = module.external_dns_irsa.iam_role_arn
}

output "vpc_cni_role_arn" {
  description = "VPC CNI IRSA role ARN"
  value       = module.vpc_cni_irsa.iam_role_arn
}
