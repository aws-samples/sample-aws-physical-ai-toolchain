# SPDX-License-Identifier: Apache-2.0

# IAM policy for S3 access
resource "aws_iam_policy" "osmo_s3_access" {
  name        = "${var.name_prefix}-osmo-s3-access"
  description = "Policy for OSMO S3 bucket access"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ListBuckets"
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [
          aws_s3_bucket.workflows.arn,
          aws_s3_bucket.datasets.arn
        ]
      },
      {
        Sid    = "ReadWriteObjects"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:GetObjectVersion",
          "s3:GetObjectTagging",
          "s3:PutObjectTagging"
        ]
        Resource = [
          "${aws_s3_bucket.workflows.arn}/*",
          "${aws_s3_bucket.datasets.arn}/*"
        ]
      },
      {
        Sid    = "EncryptDecrypt"
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = [aws_kms_key.osmo.arn]
      },
      {
        Sid    = "AllowCredentialValidation"
        Effect = "Allow"
        Action = [
          "iam:SimulatePrincipalPolicy"
        ]
        Resource = [aws_iam_user.osmo_s3.arn]
      }
    ]
  })

  tags = var.common_tags
}

# Dedicated IAM user for OSMO S3 access (OSMO does not support IRSA).
# Reuses the osmo_s3_access policy above for least-privilege S3 + KMS access.
resource "aws_iam_user" "osmo_s3" {
  name = "${var.name_prefix}-osmo-s3"
  path = "/osmo/"

  tags = merge(var.common_tags, {
    Purpose = "OSMO S3 storage credentials"
  })
}

resource "aws_iam_user_policy_attachment" "osmo_s3" {
  user       = aws_iam_user.osmo_s3.name
  policy_arn = aws_iam_policy.osmo_s3_access.arn
}

resource "aws_iam_access_key" "osmo_s3" {
  user = aws_iam_user.osmo_s3.name
}

# IAM policy for ECR access
resource "aws_iam_policy" "osmo_ecr_access" {
  name        = "${var.name_prefix}-osmo-ecr-access"
  description = "Policy for ECR image pull access"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ECRGetAuthToken"
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken"
        ]
        Resource = "*"
      },
      {
        Sid    = "ECRPullImages"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      }
    ]
  })

  tags = var.common_tags
}

# IAM policy for CloudWatch Logs
resource "aws_iam_policy" "osmo_cloudwatch_logs" {
  name        = "${var.name_prefix}-osmo-cloudwatch-logs"
  description = "Policy for CloudWatch Logs access"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams"
        ]
        Resource = [
          "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/osmo/*",
          "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/osmo/*:*"
        ]
      }
    ]
  })

  tags = var.common_tags
}
