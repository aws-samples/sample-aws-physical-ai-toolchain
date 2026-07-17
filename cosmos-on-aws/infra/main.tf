data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = var.aws_region
  prefix     = "${var.project_name}-${var.environment}"
}

# Read shared resources from foundation
data "aws_ssm_parameter" "cosmos_transfer_ecr" {
  name = "/${var.project_name}/ecr/cosmos-transfer"
}

data "aws_ssm_parameter" "cosmos3_ecr" {
  name = "/${var.project_name}/ecr/cosmos3"
}

data "aws_ssm_parameter" "cosmos_instance_profile" {
  name = "/${var.project_name}/cosmos-instance-profile"
}

# =============================================================================
# CODEBUILD: COSMOS TRANSFER 2.5
# =============================================================================

resource "aws_codebuild_project" "cosmos_transfer" {
  name         = "${local.prefix}-cosmos-transfer-build"
  description  = "Build Cosmos Transfer 2.5 container from source"
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
      value = data.aws_ssm_parameter.cosmos_transfer_ecr.value
    }

    environment_variable {
      name  = "COSMOS_REPO"
      value = var.cosmos_transfer_repo
    }

    environment_variable {
      name  = "COSMOS_REF"
      value = var.cosmos_transfer_ref
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
    buildspec = file("${path.module}/../containers/cosmos-transfer/buildspec.yml")
  }

  build_timeout = 120

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "cosmos-transfer"
  }
}

# =============================================================================
# CODEBUILD: COSMOS 3
# =============================================================================

resource "aws_codebuild_project" "cosmos3" {
  name         = "${local.prefix}-cosmos3-build"
  description  = "Build Cosmos 3 (cosmos-framework) container from source"
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
      value = data.aws_ssm_parameter.cosmos3_ecr.value
    }

    environment_variable {
      name  = "COSMOS3_REPO"
      value = var.cosmos3_repo
    }

    environment_variable {
      name  = "COSMOS3_REF"
      value = var.cosmos3_ref
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
    buildspec = file("${path.module}/../containers/cosmos3/buildspec.yml")
  }

  build_timeout = 120

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "cosmos3"
  }
}

# =============================================================================
# IAM: CODEBUILD ROLE
# =============================================================================

resource "aws_iam_role" "codebuild" {
  name = "${local.prefix}-cosmos-codebuild-role"

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
  name = "${local.prefix}-cosmos-codebuild-policy"
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
          "arn:aws:ecr:${local.region}:${local.account_id}:repository/${var.project_name}/cosmos-transfer",
          "arn:aws:ecr:${local.region}:${local.account_id}:repository/${var.project_name}/cosmos3",
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
      }
    ]
  })
}
