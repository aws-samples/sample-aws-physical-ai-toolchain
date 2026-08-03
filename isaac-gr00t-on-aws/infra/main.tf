data "aws_caller_identity" "current" {}

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

data "aws_ssm_parameter" "groot_training_ecr" {
  name = "/${var.project_name}/ecr/groot-training"
}

data "aws_ssm_parameter" "groot_inference_ecr" {
  name = "/${var.project_name}/ecr/groot-inference"
}

# =============================================================================
# CODEBUILD: GROOT TRAINING CONTAINER
# =============================================================================

resource "aws_codebuild_project" "groot_training" {
  name         = "${local.prefix}-gr00t-training-build"
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
      value = data.aws_ssm_parameter.groot_training_ecr.value
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
    buildspec = file("${path.module}/../../containers/gr00t-training/buildspec.yml")
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
  name         = "${local.prefix}-gr00t-inference-build"
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
      value = data.aws_ssm_parameter.groot_inference_ecr.value
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
    buildspec = file("${path.module}/../../containers/gr00t-inference/buildspec.yml")
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
  name = "${local.prefix}-gr00t-codebuild-role"

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
  name = "${local.prefix}-gr00t-codebuild-policy"
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
        Resource = [
          "arn:aws:ecr:${local.region}:${local.account_id}:repository/${var.project_name}/groot-training",
          "arn:aws:ecr:${local.region}:${local.account_id}:repository/${var.project_name}/groot-inference",
          "arn:aws:ecr:${local.region}:763104351884:repository/*",
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
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:GetObjectVersion", "s3:GetBucketLocation", "s3:ListBucket"]
        Resource = ["*"]
      },
      {
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = ["arn:aws:secretsmanager:*:*:secret:${var.project_name}/ngc-api-key*"]
      }
    ]
  })
}
