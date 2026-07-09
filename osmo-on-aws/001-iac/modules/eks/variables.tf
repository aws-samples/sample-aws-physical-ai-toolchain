# SPDX-License-Identifier: Apache-2.0

#------------------------------------------------------------------------------
# Core Configuration
#------------------------------------------------------------------------------

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
}

variable "common_tags" {
  description = "Common tags for all resources"
  type        = map(string)
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
}

#------------------------------------------------------------------------------
# VPC Configuration
#------------------------------------------------------------------------------

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "private_subnets" {
  description = "Private subnet IDs"
  type        = list(string)
}

#------------------------------------------------------------------------------
# EKS Cluster Configuration
#------------------------------------------------------------------------------

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
}

variable "cluster_endpoint_public_access" {
  description = "Enable public access to cluster endpoint"
  type        = bool
}

variable "cluster_endpoint_private_access" {
  description = "Enable private access to cluster endpoint"
  type        = bool
}

variable "eks_admin_principal_arns" {
  description = "IAM principal ARNs for EKS admin access"
  type        = list(string)
}

#------------------------------------------------------------------------------
# System Node Group Configuration
#------------------------------------------------------------------------------

variable "system_node_instance_types" {
  description = "Instance types for system node group"
  type        = list(string)
}

variable "system_node_min_size" {
  description = "Minimum number of system nodes"
  type        = number
}

variable "system_node_max_size" {
  description = "Maximum number of system nodes"
  type        = number
}

variable "system_node_desired_size" {
  description = "Desired number of system nodes"
  type        = number
}

#------------------------------------------------------------------------------
# GPU Node Group Configuration
#------------------------------------------------------------------------------

variable "deploy_gpu_nodes" {
  description = "Deploy GPU node group"
  type        = bool
}

variable "gpu_node_instance_types" {
  description = "Instance types for GPU node group"
  type        = list(string)
}

variable "gpu_node_min_size" {
  description = "Minimum number of GPU nodes"
  type        = number
}

variable "gpu_node_max_size" {
  description = "Maximum number of GPU nodes"
  type        = number
}

variable "gpu_node_desired_size" {
  description = "Desired number of GPU nodes"
  type        = number
}

variable "gpu_node_capacity_type" {
  description = "Capacity type for GPU nodes"
  type        = string
}

variable "gpu_ami_type" {
  description = "AMI type for GPU nodes (ignored when gpu_ami_id is set)"
  type        = string
}

variable "gpu_ami_id" {
  description = "Custom AMI ID for GPU nodes (e.g. Canonical Ubuntu EKS image). When set, gpu_ami_type is ignored and bootstrap user data is auto-generated."
  type        = string
  default     = ""
}

variable "gpu_taints" {
  description = "Taints for GPU node group"
  type = list(object({
    key    = string
    value  = string
    effect = string
  }))
}

variable "gpu_node_volume_size" {
  description = "Root EBS volume size (GiB) for GPU nodes. Must be large enough to hold container images (Isaac Sim ~30 GB, Cosmos Predict2 ~25 GB), GPU operator drivers, and the kubelet's image-gc + eviction reserves."
  type        = number
  default     = 500

  validation {
    condition     = var.gpu_node_volume_size >= 200
    error_message = "gpu_node_volume_size must be at least 200 GiB; GPU container images (Isaac Sim, Cosmos) will not fit on smaller volumes."
  }
}

#------------------------------------------------------------------------------
# IRSA Service Accounts
#------------------------------------------------------------------------------

variable "osmo_namespace" {
  description = "Kubernetes namespace for OSMO"
  type        = string
}

variable "osmo_service_account" {
  description = "Service account name for OSMO service"
  type        = string
}

variable "osmo_operator_namespace" {
  description = "Kubernetes namespace where the OSMO backend operator runs"
  type        = string
}

variable "osmo_backend_listener_sa" {
  description = "ServiceAccount name created by the backend-operator chart for the listener (release-name prefixed)"
  type        = string
}

variable "osmo_backend_worker_sa" {
  description = "ServiceAccount name created by the backend-operator chart for the worker (release-name prefixed)"
  type        = string
}

variable "osmo_workflows_namespace" {
  description = "Kubernetes namespace where OSMO workflow task pods run"
  type        = string
}

variable "osmo_workflow_sa" {
  description = "Service account name used by OSMO workflow task pods (IRSA for dataset S3 access)"
  type        = string
}

variable "aws_lb_controller_sa" {
  description = "Service account name for AWS Load Balancer Controller"
  type        = string
}

variable "cluster_autoscaler_sa" {
  description = "Service account name for Cluster Autoscaler"
  type        = string
}

variable "external_secrets_sa" {
  description = "Service account name for External Secrets Operator"
  type        = string
}

#------------------------------------------------------------------------------
# S3 Bucket ARNs
#------------------------------------------------------------------------------

variable "s3_workflows_bucket_arn" {
  description = "S3 workflows bucket ARN"
  type        = string
}

variable "s3_datasets_bucket_arn" {
  description = "S3 datasets bucket ARN"
  type        = string
}

#------------------------------------------------------------------------------
# Route53
#------------------------------------------------------------------------------

variable "route53_zone_id" {
  description = "Route53 hosted zone ID for external-dns"
  type        = string
}

#------------------------------------------------------------------------------
# Secrets Manager ARN
#------------------------------------------------------------------------------

variable "secrets_manager_arn" {
  description = "Secrets Manager ARN for OSMO secrets"
  type        = string
}

#------------------------------------------------------------------------------
# KMS Key ARN
#------------------------------------------------------------------------------

variable "kms_key_arn" {
  description = "KMS key ARN for encryption"
  type        = string
}
