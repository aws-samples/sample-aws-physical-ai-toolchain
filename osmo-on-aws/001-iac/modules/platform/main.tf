# SPDX-License-Identifier: Apache-2.0

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source                = "hashicorp/aws"
      version               = ">= 5.0"
      configuration_aliases = [aws.us_east_1]
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.0"
    }
  }
}

# This module creates shared AWS infrastructure:
# - VPC with public, private, database, and elasticache subnets
# - RDS PostgreSQL (optional)
# - ElastiCache Redis (optional)
# - S3 buckets for workflows and datasets
# - KMS key for encryption
# - Secrets Manager for storing secrets
