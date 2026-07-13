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

variable "cosmos_transfer_repo" {
  description = "GitHub repo URL for Cosmos Transfer 2.5 source"
  type        = string
  default     = "https://github.com/nvidia-cosmos/cosmos-transfer2.5.git"
}

variable "cosmos_transfer_ref" {
  description = "Git ref (tag/branch) for Cosmos Transfer build"
  type        = string
  default     = "v1.5.4"
}

variable "cosmos3_repo" {
  description = "GitHub repo URL for Cosmos 3 framework source"
  type        = string
  default     = "https://github.com/NVIDIA/cosmos-framework.git"
}

variable "cosmos3_ref" {
  description = "Git ref (tag/branch) for Cosmos 3 build"
  type        = string
  default     = "main"
}
