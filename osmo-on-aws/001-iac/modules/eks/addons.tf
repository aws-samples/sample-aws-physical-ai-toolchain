# SPDX-License-Identifier: Apache-2.0

# Additional EKS addons configuration
# Core addons (vpc-cni, coredns, kube-proxy, ebs-csi-driver) are configured in main.tf

# Note: GPU Operator, AWS Load Balancer Controller, External Secrets Operator,
# and Cluster Autoscaler are deployed via Helm in 002-setup scripts
# to provide more flexibility in configuration and to follow GitOps patterns.

# CloudWatch Container Insights (optional)
# Uncomment to enable CloudWatch logging
# resource "aws_eks_addon" "cloudwatch_observability" {
#   cluster_name  = module.eks.cluster_name
#   addon_name    = "amazon-cloudwatch-observability"
#   addon_version = "v1.5.0-eksbuild.1"
#
#   service_account_role_arn = module.cloudwatch_irsa.iam_role_arn
#
#   tags = var.common_tags
# }
#
# module "cloudwatch_irsa" {
#   source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
#   version = "~> 5.0"
#
#   role_name = "${var.name_prefix}-cloudwatch-irsa"
#
#   role_policy_arns = {
#     cloudwatch = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
#   }
#
#   oidc_providers = {
#     main = {
#       provider_arn               = module.eks.oidc_provider_arn
#       namespace_service_accounts = ["amazon-cloudwatch:cloudwatch-agent"]
#     }
#   }
#
#   tags = var.common_tags
# }
