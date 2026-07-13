data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = var.aws_region
  prefix     = "${var.project_name}-${var.environment}"
}

# Read shared resources from foundation
data "aws_ssm_parameter" "sagemaker_role_arn" {
  name = "/${var.project_name}/sagemaker-role-arn"
}

# =============================================================================
# ECR REPOSITORY
# =============================================================================

resource "aws_ecr_repository" "isaac_lab" {
  name                 = "${var.project_name}/isaac-lab"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "isaac-lab"
  }
}

resource "aws_ecr_lifecycle_policy" "isaac_lab" {
  repository = aws_ecr_repository.isaac_lab.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 5 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = { type = "expire" }
    }]
  })
}

# =============================================================================
# CODEBUILD: ISAAC LAB CONTAINER
# =============================================================================

resource "aws_codebuild_project" "isaac_lab" {
  name         = "${local.prefix}-isaac-lab-build"
  description  = "Build Isaac Lab RL training container (NGC base ~16 GB)"
  service_role = aws_iam_role.codebuild.arn

  artifacts {
    type = "NO_ARTIFACTS"
  }

  environment {
    compute_type    = "BUILD_GENERAL1_2XLARGE"
    image           = "aws/codebuild/standard:7.0"
    type            = "LINUX_CONTAINER"
    privileged_mode = true

    environment_variable {
      name  = "ECR_REPO_URI"
      value = aws_ecr_repository.isaac_lab.repository_url
    }

    environment_variable {
      name  = "AWS_ACCOUNT_ID"
      value = local.account_id
    }

    environment_variable {
      name  = "AWS_DEFAULT_REGION"
      value = local.region
    }

    # NGC API key for pulling nvcr.io/nvidia/isaac-lab base image
    dynamic "environment_variable" {
      for_each = var.ngc_secret_arn != "" ? [1] : []
      content {
        name  = "NGC_API_KEY"
        value = var.ngc_secret_arn
        type  = "SECRETS_MANAGER"
      }
    }
  }

  source {
    type      = "NO_SOURCE"
    buildspec = file("${path.module}/../containers/buildspec.yml")
  }

  build_timeout = 120

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "isaac-lab"
  }
}

# =============================================================================
# IAM: CODEBUILD ROLE
# =============================================================================

resource "aws_iam_role" "codebuild" {
  name = "${local.prefix}-isaac-lab-codebuild-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "codebuild.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "codebuild" {
  name = "${local.prefix}-isaac-lab-codebuild-policy"
  role = aws_iam_role.codebuild.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = ["*"]
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload"
        ]
        Resource = [aws_ecr_repository.isaac_lab.arn]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = ["*"]
      },
      {
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = var.ngc_secret_arn != "" ? [var.ngc_secret_arn] : ["arn:aws:secretsmanager:*:*:secret:${var.project_name}/ngc-api-key*"]
      }
    ]
  })
}
