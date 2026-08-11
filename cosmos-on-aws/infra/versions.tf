terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0, < 6.0.0" # terraform-aws-modules/eks/aws v20 caps at <6.0.0
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# helm/kubernetes providers only do anything when enable_eks_cluster=true and
# deploy Cluster Autoscaler (see eks.tf). When the cluster is disabled these
# configs point at an empty/invalid endpoint, which is fine — no resources
# from these providers get created in that case.
provider "kubernetes" {
  host                   = var.enable_eks_cluster ? module.cosmos3_eks[0].cluster_endpoint : null
  cluster_ca_certificate = var.enable_eks_cluster ? base64decode(module.cosmos3_eks[0].cluster_certificate_authority_data) : null
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = var.enable_eks_cluster ? ["eks", "get-token", "--cluster-name", module.cosmos3_eks[0].cluster_name, "--region", var.aws_region] : []
  }
}

provider "helm" {
  kubernetes {
    host                   = var.enable_eks_cluster ? module.cosmos3_eks[0].cluster_endpoint : null
    cluster_ca_certificate = var.enable_eks_cluster ? base64decode(module.cosmos3_eks[0].cluster_certificate_authority_data) : null
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = var.enable_eks_cluster ? ["eks", "get-token", "--cluster-name", module.cosmos3_eks[0].cluster_name, "--region", var.aws_region] : []
    }
  }
}
