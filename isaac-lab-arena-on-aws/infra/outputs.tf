output "ecr_repository_uris" {
  description = "Map of component ECR repository name -> repository URI."
  value       = { for name, repo in aws_ecr_repository.repos : name => repo.repository_url }
}

output "codebuild_project" {
  description = "Name of the deployed image-build CodeBuild project."
  value       = aws_codebuild_project.image_build.name
}

output "foundation_models_bucket" {
  description = "Foundation models bucket used for workload outputs; build source uses the component trust bucket."
  value       = local.models_bucket
}

output "hf_secret_policy_arn" {
  description = "ARN of the detachable managed policy granting the component training and workload roles HF-secret read."
  value       = aws_iam_policy.hf_secret_read.arn
}

# --- Gated-registration trust boundary (C4 + C5) ---
output "trust_bucket" {
  description = "Component-owned bucket holding promoted artifacts and attestations. The Foundation role has no grant on it."
  value       = aws_s3_bucket.trust.id
}

output "workload_role_arn" {
  description = "Role for SimEval. FineTune uses the separate training role. Cannot write trusted code, evidence or promoted artifacts, and cannot pass a role."
  value       = aws_iam_role.workload.arn
}

output "validation_role_arn" {
  description = "Role for Validate. Publishes promoted artifacts and attestations; cannot replace the validation code it runs."
  value       = aws_iam_role.validation.arn
}
