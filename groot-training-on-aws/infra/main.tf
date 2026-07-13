data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = var.aws_region
  prefix     = "${var.project_name}-${var.environment}"
}

# Read shared resources from foundation (via SSM)
data "aws_ssm_parameter" "datasets_bucket" {
  name = "/${var.project_name}/datasets-bucket"
}

data "aws_ssm_parameter" "sagemaker_role_arn" {
  name = "/${var.project_name}/sagemaker-role-arn"
}

# =============================================================================
# ECR REPOSITORIES
# =============================================================================

resource "aws_ecr_repository" "groot_training" {
  name                 = "${var.project_name}/groot-training"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "groot-training"
  }
}

resource "aws_ecr_lifecycle_policy" "groot_training" {
  repository = aws_ecr_repository.groot_training.name

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

resource "aws_ecr_repository" "groot_inference" {
  name                 = "${var.project_name}/groot-inference"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "groot-inference"
  }
}

resource "aws_ecr_lifecycle_policy" "groot_inference" {
  repository = aws_ecr_repository.groot_inference.name

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
# CODEBUILD: GROOT TRAINING CONTAINER
# =============================================================================

resource "aws_codebuild_project" "groot_training" {
  name         = "${local.prefix}-groot-training-build"
  description  = "Build GR00T fine-tuning container from PyTorch DLC base + Isaac-GR00T"
  service_role = aws_iam_role.codebuild.arn

  artifacts {
    type = "NO_ARTIFACTS"
  }

  environment {
    compute_type    = "BUILD_GENERAL1_LARGE"
    image           = "aws/codebuild/standard:7.0"
    type            = "LINUX_CONTAINER"
    privileged_mode = true

    environment_variable {
      name  = "ECR_REPO_URI"
      value = aws_ecr_repository.groot_training.repository_url
    }

    environment_variable {
      name  = "AWS_ACCOUNT_ID"
      value = local.account_id
    }

    environment_variable {
      name  = "AWS_DEFAULT_REGION"
      value = local.region
    }
  }

  source {
    type      = "NO_SOURCE"
    buildspec = file("${path.module}/../containers/groot-training/buildspec.yml")
  }

  build_timeout = 60

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "groot-training"
  }
}

# =============================================================================
# CODEBUILD: GROOT INFERENCE CONTAINER
# =============================================================================

resource "aws_codebuild_project" "groot_inference" {
  name         = "${local.prefix}-groot-inference-build"
  description  = "Build GR00T inference/serving container"
  service_role = aws_iam_role.codebuild.arn

  artifacts {
    type = "NO_ARTIFACTS"
  }

  environment {
    compute_type    = "BUILD_GENERAL1_LARGE"
    image           = "aws/codebuild/standard:7.0"
    type            = "LINUX_CONTAINER"
    privileged_mode = true

    environment_variable {
      name  = "ECR_REPO_URI"
      value = aws_ecr_repository.groot_inference.repository_url
    }

    environment_variable {
      name  = "AWS_ACCOUNT_ID"
      value = local.account_id
    }

    environment_variable {
      name  = "AWS_DEFAULT_REGION"
      value = local.region
    }
  }

  source {
    type      = "NO_SOURCE"
    buildspec = file("${path.module}/../containers/groot-inference/buildspec.yml")
  }

  build_timeout = 60

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "groot-inference"
  }
}

# =============================================================================
# IAM: CODEBUILD ROLE
# =============================================================================

resource "aws_iam_role" "codebuild" {
  name = "${local.prefix}-groot-codebuild-role"

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
  name = "${local.prefix}-groot-codebuild-policy"
  role = aws_iam_role.codebuild.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken"
        ]
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
        Resource = [
          aws_ecr_repository.groot_training.arn,
          aws_ecr_repository.groot_inference.arn,
        ]
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
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetBucketLocation"
        ]
        Resource = ["*"]
      }
    ]
  })
}
