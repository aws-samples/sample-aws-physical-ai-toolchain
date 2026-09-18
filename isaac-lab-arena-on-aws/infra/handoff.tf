# Evaluation handoff storage (cycle-7 C3 / cycle-8 C2).
#
# The problem: SimEval wrote its result archive into the SHARED Foundation models bucket, and
# Validate consumed it through a step-property S3 URI with no version. The Foundation SageMaker
# role holds PutObject and DeleteObject across that whole bucket, and the documented combined
# toolchain has peer components submitting jobs under that role. So a peer worker could replace
# a completed SimEval archive between SimEval finishing and Validate downloading, preserving the
# genuine checkpoint identity while substituting the score. Every check Validate performs
# authenticates the MODEL BYTES; none of them authenticated the ORIGIN OF THE RESULT.
#
# A structural step-property reference establishes which KEY belongs to SimEval. It does not
# establish that the bytes later occupying that key are SimEval's.
#
# Why a separate bucket rather than the existing trust bucket: the workload role's identity
# policy denies writes across the ENTIRE trust bucket, not just code/v1/*. Allowing SimEval to
# write there would require narrowing that bucket-wide denial, which is the property the trust
# bucket exists to provide. Raw evaluator output and validation-approved evidence are different
# trust levels and belong in different storage.
#
# SCOPE OF THIS CONTROL -- read before relying on it. This closes the direct S3 replacement
# path. It does NOT contain a compromised principal holding the full Foundation role, because
# Foundation attaches AmazonSageMakerFullAccess, which grants iam:PassRole to SageMaker for
# arbitrary roles: such a principal can launch a job AS the component's own workload role and
# write these namespaces legitimately. Closing that requires moving peer components onto
# restricted worker identities, or removing Foundation's ability to impersonate component roles.
# Both are account-level decisions outside this component. trust_boundary.tf already records
# that Foundation holders are workflow administrators.
resource "aws_s3_bucket" "handoff" {
  bucket = "${local.prefix}-handoff-${local.account_id}"

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Purpose     = "evaluation-evidence-handoff"
  }
}

# Versioning is what makes a replacement detectable rather than merely denied: a client that
# records the VersionId it produced can later read that exact version back.
resource "aws_s3_bucket_versioning" "handoff" {
  bucket = aws_s3_bucket.handoff.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "handoff" {
  bucket                  = aws_s3_bucket.handoff.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "handoff" {
  bucket = aws_s3_bucket.handoff.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

locals {
  # Raw evaluator output. SimEval's workload role is the ONLY permitted publisher.
  handoff_eval_arns = ["${aws_s3_bucket.handoff.arn}/eval/v1/*"]
  # The gate's own receipt. The validation role is the ONLY permitted publisher, so a worker
  # cannot manufacture a validated result either.
  handoff_validated_arns = ["${aws_s3_bucket.handoff.arn}/validated/v1/*"]
}

data "aws_iam_policy_document" "handoff_bucket" {
  # I2: merged here rather than as a second aws_s3_bucket_policy -- two bucket policies on
  # one bucket silently overwrite each other, so the gate fix would have removed the
  # publisher restrictions it sits beside.
  source_policy_documents = [data.aws_iam_policy_document.handoff_pipeline_read.json]

  # Only the evaluation identity may publish raw evidence. Naming the denied principals
  # explicitly rather than using a NotPrincipal Allow: NotPrincipal with Deny is notoriously
  # easy to get wrong, and these are the identities that actually exist in this account's
  # workflow.
  statement {
    sid       = "OnlyWorkloadMayPublishRawEvidence"
    effect    = "Deny"
    actions   = ["s3:PutObject", "s3:DeleteObject", "s3:DeleteObjectVersion", "s3:PutObjectAcl"]
    resources = local.handoff_eval_arns
    principals {
      type        = "AWS"
      identifiers = ["*"]
    }
    condition {
      test     = "ArnNotEquals"
      variable = "aws:PrincipalArn"
      values   = [aws_iam_role.workload.arn]
    }
  }

  # Only the validation identity may publish the gate's receipt.
  statement {
    sid       = "OnlyValidationMayPublishReceipts"
    effect    = "Deny"
    actions   = ["s3:PutObject", "s3:DeleteObject", "s3:DeleteObjectVersion", "s3:PutObjectAcl"]
    resources = local.handoff_validated_arns
    principals {
      type        = "AWS"
      identifiers = ["*"]
    }
    condition {
      test     = "ArnNotEquals"
      variable = "aws:PrincipalArn"
      values   = [aws_iam_role.validation.arn]
    }
  }

  # No principal deletes evidence, including the roles that wrote it. A deletion followed by a
  # fresh write by the same role would otherwise be an allowed replacement.
  statement {
    sid    = "NobodyDeletesHandoffEvidence"
    effect = "Deny"
    actions = [
      "s3:DeleteObject",
      "s3:DeleteObjectVersion",
    ]
    resources = concat(local.handoff_eval_arns, local.handoff_validated_arns)
    principals {
      type        = "AWS"
      identifiers = ["*"]
    }
  }

  statement {
    sid       = "DenyUnencryptedTransport"
    effect    = "Deny"
    actions   = ["s3:*"]
    resources = [aws_s3_bucket.handoff.arn, "${aws_s3_bucket.handoff.arn}/*"]
    principals {
      type        = "AWS"
      identifiers = ["*"]
    }
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "handoff" {
  bucket = aws_s3_bucket.handoff.id
  policy = data.aws_iam_policy_document.handoff_bucket.json

  # The policy references the workload and validation roles, which must exist first.
  depends_on = [aws_iam_role.workload, aws_iam_role.validation]
}

# Published through the same discovery contract as the roles and the trust bucket, so the
# launchers cannot silently fall back to the shared bucket this replaces.
resource "aws_ssm_parameter" "handoff_bucket" {
  name  = "/${var.project_name}/component/handoff-bucket"
  type  = "String"
  value = aws_s3_bucket.handoff.id
  tags  = local.tags
}

# I11: a failed multipart publication leaves parts that are billable but invisible to a normal
# object listing. The abort path is best-effort by nature -- it runs while another failure is
# already in progress -- so storage must not depend on it succeeding. Seven days is well past
# any legitimate retry window and costs nothing for uploads that completed.
resource "aws_s3_bucket_lifecycle_configuration" "handoff_abort_incomplete" {
  bucket = aws_s3_bucket.handoff.id
  rule {
    id     = "abort-incomplete-multipart"
    status = "Enabled"
    filter {}
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# I2: SuccessGate's ConditionStep reads Validate's property file with JsonGet, and JsonGet is
# performed by the PIPELINE EXECUTION role -- the Foundation role the pipeline is upserted under --
# not by Validate's own role. Moving the receipt into this bucket therefore broke the gate: every
# job would complete successfully and the pipeline would then fail consuming the result, after all
# the capacity had been paid for.
#
# The worker-role split deliberately does not confer worker permissions on the orchestrator, so
# this grant has to be explicit. Read-only, and scoped to the receipt namespace alone: the
# orchestrator needs to read one small JSON document, not the evidence corpus.
data "aws_iam_policy_document" "handoff_pipeline_read" {
  statement {
    sid    = "PipelineExecutionRoleReadsValidatedReceipts"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:GetObjectVersion",
    ]
    resources = local.handoff_validated_arns
    principals {
      type        = "AWS"
      identifiers = [local.foundation_role_arn]
    }
  }
}
