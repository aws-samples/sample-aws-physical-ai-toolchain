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
  description = "Project name prefix"
  type        = string
  default     = "physical-ai"
}

variable "instance_type" {
  description = "EC2 instance type for Isaac Sim workstation"
  type        = string
  default     = "g6e.4xlarge"
}

variable "ami_id" {
  description = "AMI ID for Isaac Sim (use the NVIDIA Isaac Sim Marketplace AMI for your region)"
  type        = string
}

variable "key_name" {
  description = "EC2 key pair name for SSH access (optional — DCV uses browser)"
  type        = string
  default     = ""
}

variable "allowed_cidrs" {
  description = "CIDR blocks allowed to access DCV (port 8443) and SSH (port 22)"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "volume_size" {
  description = "Root EBS volume size in GB"
  type        = number
  default     = 512
}

variable "subnet_id" {
  description = "Subnet ID to launch into (must be public for DCV access). Leave empty to use default VPC."
  type        = string
  default     = ""
}
