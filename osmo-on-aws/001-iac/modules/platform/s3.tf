# SPDX-License-Identifier: Apache-2.0

# S3 bucket for workflows
resource "aws_s3_bucket" "workflows" {
  bucket        = var.s3_workflows_bucket
  force_destroy = var.s3_force_destroy

  tags = merge(var.common_tags, {
    Name    = var.s3_workflows_bucket
    Purpose = "osmo-workflows"
  })
}

resource "aws_s3_bucket_versioning" "workflows" {
  bucket = aws_s3_bucket.workflows.id

  versioning_configuration {
    status = var.s3_versioning_enabled ? "Enabled" : "Disabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "workflows" {
  bucket = aws_s3_bucket.workflows.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.osmo.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "workflows" {
  bucket = aws_s3_bucket.workflows.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "workflows" {
  bucket = aws_s3_bucket.workflows.id

  rule {
    id     = "cleanup-incomplete-uploads"
    status = "Enabled"
    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  rule {
    id     = "transition-to-intelligent-tiering"
    status = "Enabled"
    filter {}

    transition {
      days          = 30
      storage_class = "INTELLIGENT_TIERING"
    }
  }
}

# S3 bucket for datasets
resource "aws_s3_bucket" "datasets" {
  bucket        = var.s3_datasets_bucket
  force_destroy = var.s3_force_destroy

  tags = merge(var.common_tags, {
    Name    = var.s3_datasets_bucket
    Purpose = "osmo-datasets"
  })
}

resource "aws_s3_bucket_versioning" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  versioning_configuration {
    status = var.s3_versioning_enabled ? "Enabled" : "Disabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.osmo.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  rule {
    id     = "cleanup-incomplete-uploads"
    status = "Enabled"
    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  rule {
    id     = "transition-to-intelligent-tiering"
    status = "Enabled"
    filter {}

    transition {
      days          = 30
      storage_class = "INTELLIGENT_TIERING"
    }
  }
}

# CORS configuration for datasets bucket (needed for direct uploads from UI)
resource "aws_s3_bucket_cors_configuration" "datasets" {
  bucket = aws_s3_bucket.datasets.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "PUT", "POST", "HEAD"]
    # Use specified origins in production, fallback to wildcard for dev
    allowed_origins = length(var.s3_allowed_origins) > 0 ? var.s3_allowed_origins : ["*"]
    expose_headers  = ["ETag", "x-amz-meta-custom-header"]
    max_age_seconds = 3600
  }
}
