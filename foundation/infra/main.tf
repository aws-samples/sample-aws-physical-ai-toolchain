data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = var.aws_region
  prefix     = "${var.project_name}-${var.environment}"
}

# =============================================================================
# S3 BUCKETS
# =============================================================================

resource "aws_s3_bucket" "datasets" {
  bucket = "${local.prefix}-datasets-${local.account_id}"

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Purpose     = "Training datasets (LeRobot format)"
  }
}

resource "aws_s3_bucket_versioning" "datasets" {
  bucket = aws_s3_bucket.datasets.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "datasets" {
  bucket                  = aws_s3_bucket.datasets.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket" "models" {
  bucket = "${local.prefix}-models-${local.account_id}"

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Purpose     = "Trained models (ONNX, TensorRT, model.tar.gz)"
  }
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket                  = aws_s3_bucket.models.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket" "checkpoints" {
  bucket = "${local.prefix}-checkpoints-${local.account_id}"

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Purpose     = "Training checkpoints (auto-expire after 14 days)"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "checkpoints" {
  bucket = aws_s3_bucket.checkpoints.id

  rule {
    id     = "expire-checkpoints"
    status = "Enabled"
    expiration {
      days = 14
    }
  }
}

resource "aws_s3_bucket_public_access_block" "checkpoints" {
  bucket                  = aws_s3_bucket.checkpoints.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# =============================================================================
# IAM: SAGEMAKER EXECUTION ROLE
# =============================================================================

resource "aws_iam_role" "sagemaker" {
  name = "${local.prefix}-sagemaker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_iam_role_policy_attachment" "sagemaker_full" {
  role       = aws_iam_role.sagemaker.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy" "sagemaker_s3" {
  name = "${local.prefix}-sagemaker-s3"
  role = aws_iam_role.sagemaker.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ]
      Resource = [
        aws_s3_bucket.datasets.arn,
        "${aws_s3_bucket.datasets.arn}/*",
        aws_s3_bucket.models.arn,
        "${aws_s3_bucket.models.arn}/*",
        aws_s3_bucket.checkpoints.arn,
        "${aws_s3_bucket.checkpoints.arn}/*",
      ]
    }]
  })
}

resource "aws_iam_role_policy" "sagemaker_ecr" {
  name = "${local.prefix}-sagemaker-ecr"
  role = aws_iam_role.sagemaker.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ]
      Resource = ["*"]
    }]
  })
}

# =============================================================================
# IAM: COSMOS EC2 INSTANCE ROLE
# =============================================================================

resource "aws_iam_role" "cosmos" {
  name = "${local.prefix}-cosmos-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_iam_role_policy_attachment" "cosmos_ssm" {
  role       = aws_iam_role.cosmos.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy_attachment" "cosmos_ecr" {
  role       = aws_iam_role.cosmos.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_role_policy" "cosmos_secrets" {
  name = "${local.prefix}-cosmos-secrets"
  role = aws_iam_role.cosmos.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["secretsmanager:GetSecretValue"]
      Resource = [
        "arn:aws:secretsmanager:${local.region}:${local.account_id}:secret:${var.project_name}/ngc-api-key*",
        "arn:aws:secretsmanager:${local.region}:${local.account_id}:secret:${var.project_name}/hf-token*",
        "arn:aws:secretsmanager:${local.region}:${local.account_id}:secret:${var.project_name}/nim-api-key*",
      ]
    }]
  })
}

resource "aws_iam_role_policy" "cosmos_s3" {
  name = "${local.prefix}-cosmos-s3"
  role = aws_iam_role.cosmos.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ]
      Resource = [
        aws_s3_bucket.datasets.arn,
        "${aws_s3_bucket.datasets.arn}/*",
        aws_s3_bucket.models.arn,
        "${aws_s3_bucket.models.arn}/*",
      ]
    }]
  })
}

resource "aws_iam_instance_profile" "cosmos" {
  name = "${local.prefix}-cosmos-profile"
  role = aws_iam_role.cosmos.name
}

# =============================================================================
# SSM PARAMETERS (shared outputs for other components to reference)
# =============================================================================

resource "aws_ssm_parameter" "datasets_bucket" {
  name  = "/${var.project_name}/datasets-bucket"
  type  = "String"
  value = aws_s3_bucket.datasets.bucket
}

resource "aws_ssm_parameter" "models_bucket" {
  name  = "/${var.project_name}/models-bucket"
  type  = "String"
  value = aws_s3_bucket.models.bucket
}

resource "aws_ssm_parameter" "checkpoints_bucket" {
  name  = "/${var.project_name}/checkpoints-bucket"
  type  = "String"
  value = aws_s3_bucket.checkpoints.bucket
}

resource "aws_ssm_parameter" "sagemaker_role_arn" {
  name  = "/${var.project_name}/sagemaker-role-arn"
  type  = "String"
  value = aws_iam_role.sagemaker.arn
}

resource "aws_ssm_parameter" "cosmos_instance_profile" {
  name  = "/${var.project_name}/cosmos-instance-profile"
  type  = "String"
  value = aws_iam_instance_profile.cosmos.name
}
