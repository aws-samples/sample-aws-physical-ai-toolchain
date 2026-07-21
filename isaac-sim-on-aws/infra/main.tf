data "aws_caller_identity" "current" {}

locals {
  prefix = "${var.project_name}-${var.environment}"
}

# =============================================================================
# SECURITY GROUP
# =============================================================================

resource "aws_security_group" "workstation" {
  name        = "${local.prefix}-isaac-sim-sg"
  description = "Security group for Isaac Sim workstation (DCV + SSH)"

  # DCV (remote desktop via browser)
  ingress {
    from_port   = 8443
    to_port     = 8443
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidrs
    description = "NICE DCV (HTTPS)"
  }

  # SSH (optional)
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidrs
    description = "SSH"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound"
  }

  tags = {
    Name        = "${local.prefix}-isaac-sim-sg"
    Project     = var.project_name
    Environment = var.environment
  }
}

# =============================================================================
# IAM ROLE (SSM + ECR access)
# =============================================================================

resource "aws_iam_role" "workstation" {
  name = "${local.prefix}-isaac-sim-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "workstation_ssm" {
  role       = aws_iam_role.workstation.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy_attachment" "workstation_ecr" {
  role       = aws_iam_role.workstation.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_instance_profile" "workstation" {
  name = "${local.prefix}-isaac-sim-profile"
  role = aws_iam_role.workstation.name
}

# =============================================================================
# EC2 INSTANCE
# =============================================================================

resource "aws_instance" "workstation" {
  ami                    = var.ami_id
  instance_type          = var.instance_type
  key_name               = var.key_name != "" ? var.key_name : null
  vpc_security_group_ids = [aws_security_group.workstation.id]
  iam_instance_profile   = aws_iam_instance_profile.workstation.name
  subnet_id              = var.subnet_id != "" ? var.subnet_id : null

  root_block_device {
    volume_size = var.volume_size
    volume_type = "gp3"
    encrypted   = true
  }

  metadata_options {
    http_tokens = "required" # IMDSv2
  }

  tags = {
    Name        = "${local.prefix}-isaac-sim"
    Project     = var.project_name
    Environment = var.environment
    Component   = "isaac-sim-workstation"
  }
}
