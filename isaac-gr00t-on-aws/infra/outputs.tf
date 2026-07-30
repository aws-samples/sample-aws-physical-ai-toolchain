output "groot_training_ecr_uri" {
  description = "ECR URI for the GR00T training container"
  value       = data.aws_ssm_parameter.groot_training_ecr.value
  sensitive   = true
}

output "groot_inference_ecr_uri" {
  description = "ECR URI for the GR00T inference container"
  value       = data.aws_ssm_parameter.groot_inference_ecr.value
  sensitive   = true
}

output "codebuild_training_project" {
  description = "CodeBuild project name for GR00T training container"
  value       = aws_codebuild_project.groot_training.name
}

output "codebuild_inference_project" {
  description = "CodeBuild project name for GR00T inference container"
  value       = aws_codebuild_project.groot_inference.name
}
