# SPDX-License-Identifier: Apache-2.0

# OSMO Service IRSA role
module "osmo_service_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "${var.name_prefix}-osmo-service-irsa"

  role_policy_arns = {
    s3_access = aws_iam_policy.osmo_service_s3.arn
    secrets   = aws_iam_policy.osmo_service_secrets.arn
    kms       = aws_iam_policy.osmo_service_kms.arn
  }

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["${var.osmo_namespace}:${var.osmo_service_account}"]
    }
  }

  tags = var.common_tags
}

# OSMO Service S3 policy
resource "aws_iam_policy" "osmo_service_s3" {
  name        = "${var.name_prefix}-osmo-service-s3"
  description = "S3 access policy for OSMO service"

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
          var.s3_workflows_bucket_arn,
          var.s3_datasets_bucket_arn
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
          "${var.s3_workflows_bucket_arn}/*",
          "${var.s3_datasets_bucket_arn}/*"
        ]
      }
    ]
  })

  tags = var.common_tags
}

# OSMO Service Secrets Manager policy
resource "aws_iam_policy" "osmo_service_secrets" {
  name        = "${var.name_prefix}-osmo-service-secrets"
  description = "Secrets Manager access policy for OSMO service"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadSecrets"
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = ["${var.secrets_manager_arn}*"]
      }
    ]
  })

  tags = var.common_tags
}

# OSMO Service KMS policy
resource "aws_iam_policy" "osmo_service_kms" {
  name        = "${var.name_prefix}-osmo-service-kms"
  description = "KMS access policy for OSMO service"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "DecryptSecrets"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey",
          "kms:GenerateDataKey"
        ]
        Resource = [var.kms_key_arn]
      }
    ]
  })

  tags = var.common_tags
}

# OSMO Backend Operator IRSA role
module "osmo_backend_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "${var.name_prefix}-osmo-backend-irsa"

  role_policy_arns = {
    s3_access = aws_iam_policy.osmo_backend_s3.arn
    secrets   = aws_iam_policy.osmo_backend_secrets.arn
    kms       = aws_iam_policy.osmo_backend_kms.arn
    ecr       = aws_iam_policy.osmo_backend_ecr.arn
  }

  # The backend-operator chart creates two release-name-prefixed SAs
  # (<release>-backend-listener / <release>-backend-worker) in the operator
  # namespace, and does NOT expose SA annotations — so the trust policy must
  # name those SAs directly (annotation is applied via kubectl in
  # 05-deploy-osmo-backend.sh).
  oidc_providers = {
    main = {
      provider_arn = module.eks.oidc_provider_arn
      namespace_service_accounts = [
        "${var.osmo_operator_namespace}:${var.osmo_backend_listener_sa}",
        "${var.osmo_operator_namespace}:${var.osmo_backend_worker_sa}",
      ]
    }
  }

  tags = var.common_tags
}

# OSMO Backend S3 policy
resource "aws_iam_policy" "osmo_backend_s3" {
  name        = "${var.name_prefix}-osmo-backend-s3"
  description = "S3 access policy for OSMO backend operator"

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
          var.s3_workflows_bucket_arn,
          var.s3_datasets_bucket_arn
        ]
      },
      {
        Sid    = "ReadWriteObjects"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:GetObjectVersion"
        ]
        Resource = [
          "${var.s3_workflows_bucket_arn}/*",
          "${var.s3_datasets_bucket_arn}/*"
        ]
      }
    ]
  })

  tags = var.common_tags
}

# OSMO Backend Secrets Manager policy
resource "aws_iam_policy" "osmo_backend_secrets" {
  name        = "${var.name_prefix}-osmo-backend-secrets"
  description = "Secrets Manager access policy for OSMO backend operator"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadSecrets"
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = ["${var.secrets_manager_arn}*"]
      }
    ]
  })

  tags = var.common_tags
}

# OSMO Backend KMS policy
resource "aws_iam_policy" "osmo_backend_kms" {
  name        = "${var.name_prefix}-osmo-backend-kms"
  description = "KMS access policy for OSMO backend operator"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "DecryptSecrets"
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
          "kms:DescribeKey",
          "kms:GenerateDataKey"
        ]
        Resource = [var.kms_key_arn]
      }
    ]
  })

  tags = var.common_tags
}

# OSMO Backend ECR policy
resource "aws_iam_policy" "osmo_backend_ecr" {
  name        = "${var.name_prefix}-osmo-backend-ecr"
  description = "ECR access policy for OSMO backend operator"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ECRGetAuthToken"
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
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

# OSMO Workflow-pod IRSA role
# ---------------------------------------------------------------------------
# Workflow task pods (Isaac Sim / Cosmos containers + the osmo-ctrl sidecar)
# run in the workflows namespace and perform dataset read/write directly.
# Binding their ServiceAccount to this role lets osmo-ctrl authenticate to S3
# via the IRSA web-identity token (boto3 default credential chain) instead of
# falling back to the EKS node instance role (which is denied). This removes
# the need to register static S3 access keys as OSMO dataset credentials.
module "osmo_workflow_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "${var.name_prefix}-osmo-workflow-irsa"

  role_policy_arns = {
    s3_access = aws_iam_policy.osmo_workflow_s3.arn
  }

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["${var.osmo_workflows_namespace}:${var.osmo_workflow_sa}"]
    }
  }

  tags = var.common_tags
}

# OSMO Workflow-pod S3 + KMS policy
resource "aws_iam_policy" "osmo_workflow_s3" {
  name        = "${var.name_prefix}-osmo-workflow-s3"
  description = "S3 access policy for OSMO workflow task pods (dataset I/O)"

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
          var.s3_workflows_bucket_arn,
          var.s3_datasets_bucket_arn
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
          "${var.s3_workflows_bucket_arn}/*",
          "${var.s3_datasets_bucket_arn}/*"
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
        Resource = [var.kms_key_arn]
      }
    ]
  })

  tags = var.common_tags
}

# AWS Load Balancer Controller IRSA role
module "aws_lb_controller_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name                              = "${var.name_prefix}-aws-lb-controller-irsa"
  attach_load_balancer_controller_policy = true

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:${var.aws_lb_controller_sa}"]
    }
  }

  tags = var.common_tags
}

# Cluster Autoscaler IRSA role
module "cluster_autoscaler_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name                        = "${var.name_prefix}-cluster-autoscaler-irsa"
  attach_cluster_autoscaler_policy = true
  cluster_autoscaler_cluster_names = [var.cluster_name]

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:${var.cluster_autoscaler_sa}"]
    }
  }

  tags = var.common_tags
}

# external-dns IRSA role
module "external_dns_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name                     = "${var.name_prefix}-external-dns-irsa"
  attach_external_dns_policy    = true
  external_dns_hosted_zone_arns = ["arn:aws:route53:::hostedzone/${var.route53_zone_id}"]

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:external-dns"]
    }
  }

  tags = var.common_tags
}

# External Secrets Operator IRSA role
module "external_secrets_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name                             = "${var.name_prefix}-external-secrets-irsa"
  attach_external_secrets_policy        = true
  external_secrets_secrets_manager_arns = ["${var.secrets_manager_arn}*"]
  external_secrets_kms_key_arns         = [var.kms_key_arn]

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["external-secrets:${var.external_secrets_sa}"]
    }
  }

  tags = var.common_tags
}
