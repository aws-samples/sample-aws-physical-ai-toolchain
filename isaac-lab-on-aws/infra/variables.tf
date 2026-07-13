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
  description = "Project name prefix (must match foundation)"
  type        = string
  default     = "physical-ai"
}

variable "ngc_secret_arn" {
  description = "Secrets Manager ARN for NGC API key (needed to pull Isaac Lab base image)"
  type        = string
  default     = ""
}
