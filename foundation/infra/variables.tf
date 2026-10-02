variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name prefix for all resources"
  type        = string
  default     = "physical-ai"
}

variable "dlc_account_id" {
  description = "AWS account hosting the Deep Learning Container images SageMaker jobs start from. A variable rather than a literal because the DLC registry account differs by Region - a deployment outside the default Region needs the matching account or ECR pulls fail with AccessDenied."
  type        = string
  default     = "763104351884"
}
