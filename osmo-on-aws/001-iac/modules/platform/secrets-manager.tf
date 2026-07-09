# SPDX-License-Identifier: Apache-2.0

# Main OSMO secrets container
resource "aws_secretsmanager_secret" "osmo" {
  name                    = "${var.name_prefix}/osmo/config"
  description             = "OSMO configuration secrets for ${var.name_prefix}"
  recovery_window_in_days = var.secrets_recovery_window_days
  kms_key_id              = aws_kms_key.osmo.arn

  tags = merge(var.common_tags, {
    Name = "${var.name_prefix}-osmo-secrets"
  })
}

# Initial secret version with placeholder values (populated out-of-band if used)
resource "aws_secretsmanager_secret_version" "osmo" {
  secret_id = aws_secretsmanager_secret.osmo.id
  secret_string = jsonencode({
    # OSMO admin credentials
    admin_username = "admin"
    admin_password = "" # Placeholder

    # OSMO API token
    api_token = "" # Placeholder

    # Master Encryption Key (MEK)
    mek_key = "" # Placeholder
    mek_kid = "key1"

    # OAuth/OIDC configuration (if applicable)
    oauth_client_id     = ""
    oauth_client_secret = ""
  })

  lifecycle {
    ignore_changes = [
      secret_string, # Allow external updates
    ]
  }
}

# Secret for NGC API key (for pulling NVIDIA images)
resource "aws_secretsmanager_secret" "ngc_api_key" {
  name                    = "${var.name_prefix}/ngc/api-key"
  description             = "NGC API key for NVIDIA container registry access"
  recovery_window_in_days = var.secrets_recovery_window_days
  kms_key_id              = aws_kms_key.osmo.arn

  tags = merge(var.common_tags, {
    Name = "${var.name_prefix}-ngc-api-key"
  })
}

resource "aws_secretsmanager_secret_version" "ngc_api_key" {
  secret_id = aws_secretsmanager_secret.ngc_api_key.id
  secret_string = jsonencode({
    api_key  = "" # Placeholder
    username = "$oauthtoken"
  })

  lifecycle {
    ignore_changes = [
      secret_string, # Allow external updates
    ]
  }
}

# S3 credentials for OSMO (IAM user access keys)
resource "aws_secretsmanager_secret" "osmo_s3_credentials" {
  name                    = "${var.name_prefix}/osmo/s3-credentials"
  description             = "OSMO S3 access keys (IAM user) for ${var.name_prefix}"
  recovery_window_in_days = var.secrets_recovery_window_days
  kms_key_id              = aws_kms_key.osmo.arn

  tags = merge(var.common_tags, {
    Name = "${var.name_prefix}-osmo-s3-credentials"
  })
}

resource "aws_secretsmanager_secret_version" "osmo_s3_credentials" {
  secret_id = aws_secretsmanager_secret.osmo_s3_credentials.id
  secret_string = jsonencode({
    access_key_id     = aws_iam_access_key.osmo_s3.id
    secret_access_key = aws_iam_access_key.osmo_s3.secret
  })
}

# IAM policy for reading OSMO secrets
resource "aws_iam_policy" "osmo_secrets_read" {
  name        = "${var.name_prefix}-osmo-secrets-read"
  description = "Policy for reading OSMO secrets from Secrets Manager"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadOSMOSecrets"
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = [
          aws_secretsmanager_secret.osmo.arn,
          aws_secretsmanager_secret.ngc_api_key.arn,
          aws_secretsmanager_secret.osmo_s3_credentials.arn,
          var.deploy_postgresql ? aws_secretsmanager_secret.rds_password[0].arn : "",
          var.deploy_redis ? aws_secretsmanager_secret.redis_auth_token[0].arn : ""
        ]
      },
      {
        Sid    = "DecryptSecrets"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey"
        ]
        Resource = [aws_kms_key.osmo.arn]
      }
    ]
  })

  tags = var.common_tags
}
