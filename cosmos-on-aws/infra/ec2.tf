# =============================================================================
# EC2: COSMOS 3 GENERATION SERVER (p5.48xlarge, 8x H100)
# =============================================================================
#
# Launches into an already-ACTIVE EC2 Capacity Block. Capacity Block purchase
# itself is NOT managed by Terraform — it's an async external reservation
# (minutes to days to become active) that doesn't fit a `terraform apply`
# lifecycle. Purchase it first (see ec2-deployment-guide.md Step 1), then set
# capacity_reservation_id below and apply.
#
# Toggle with var.enable_ec2_server (default false) so `terraform apply` is
# safe to run before you have an active Capacity Block.

variable "enable_ec2_server" {
  description = "Deploy the Cosmos 3 EC2 generation server. Requires an ACTIVE Capacity Block."
  type        = bool
  default     = false
}

variable "capacity_reservation_id" {
  description = "ID of an ACTIVE EC2 Capacity Block to launch into (e.g. cr-0123456789abcdef0). Purchase separately — see Step 1 of ec2-deployment-guide.md."
  type        = string
  default     = ""
}

variable "availability_zone" {
  description = "AZ of the Capacity Block (must match its reserved AZ)"
  type        = string
  default     = ""
}

variable "server_instance_type" {
  description = "Instance type for the Cosmos 3 server (must match the Capacity Block's reserved type)"
  type        = string
  default     = "p5.48xlarge"
}

variable "server_ami_id" {
  description = "AMI for the Cosmos 3 server (Deep Learning Base OSS Nvidia Driver GPU AMI recommended)"
  type        = string
  default     = ""
}

variable "server_volume_size_gb" {
  description = "Root EBS volume size in GB (needs room for ~30GB container + ~126GB model weights)"
  type        = number
  default     = 500
}

data "aws_ssm_parameter" "vpc_id" {
  count = var.enable_ec2_server ? 1 : 0
  name  = "/${var.project_name}/vpc-id"
}

data "aws_ssm_parameter" "private_subnet_ids" {
  count = var.enable_ec2_server ? 1 : 0
  name  = "/${var.project_name}/private-subnet-ids"
}

# Pick the private subnet in the same AZ as the Capacity Block.
data "aws_subnet" "server" {
  count = var.enable_ec2_server ? 1 : 0
  filter {
    name   = "subnet-id"
    values = split(",", data.aws_ssm_parameter.private_subnet_ids[0].value)
  }
  availability_zone = var.availability_zone
}

# Latest Deep Learning Base OSS Nvidia Driver GPU AMI, unless pinned explicitly.
data "aws_ami" "dlami" {
  count       = var.enable_ec2_server && var.server_ami_id == "" ? 1 : 0
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"]
  }
}

locals {
  server_ami_id = var.server_ami_id != "" ? var.server_ami_id : (
    var.enable_ec2_server ? data.aws_ami.dlami[0].image_id : ""
  )
}

resource "aws_security_group" "cosmos_server" {
  count       = var.enable_ec2_server ? 1 : 0
  name        = "${local.prefix}-cosmos-server-sg"
  description = "Cosmos 3 generation server - outbound only (SSM, no inbound needed)"
  vpc_id      = data.aws_ssm_parameter.vpc_id[0].value

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound (ECR, HuggingFace, S3)"
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "cosmos3-server"
  }
}

resource "aws_instance" "cosmos_server" {
  count                  = var.enable_ec2_server ? 1 : 0
  ami                    = local.server_ami_id
  instance_type          = var.server_instance_type
  subnet_id              = data.aws_subnet.server[0].id
  vpc_security_group_ids = [aws_security_group.cosmos_server[0].id]
  iam_instance_profile   = data.aws_ssm_parameter.cosmos_instance_profile.value

  # Launch into the Capacity Block reserved for this AZ/instance type.
  capacity_reservation_specification {
    capacity_reservation_target {
      capacity_reservation_id = var.capacity_reservation_id
    }
  }

  # Capacity Blocks require this market type at launch.
  instance_market_options {
    market_type = "capacity-block"
  }

  root_block_device {
    volume_size = var.server_volume_size_gb
    volume_type = "gp3"
    encrypted   = true
  }

  metadata_options {
    http_tokens = "required" # IMDSv2
  }

  tags = {
    Name        = "${local.prefix}-cosmos3-server"
    Project     = var.project_name
    Environment = var.environment
    Component   = "cosmos3-server"
  }
}

output "server_instance_id" {
  description = "Instance ID of the Cosmos 3 generation server (if enabled)"
  value       = var.enable_ec2_server ? aws_instance.cosmos_server[0].id : null
}

output "server_private_ip" {
  description = "Private IP of the Cosmos 3 generation server (if enabled)"
  value       = var.enable_ec2_server ? aws_instance.cosmos_server[0].private_ip : null
}
