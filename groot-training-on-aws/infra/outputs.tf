output "groot_training_ecr_uri" {
  description = "ECR URI for the GR00T training container"
  value       = aws_ecr_repository.groot_training.repository_url
}

output "groot_inference_ecr_uri" {
  description = "ECR URI for the GR00T inference container"
  value       = aws_ecr_repository.groot_inference.repository_url
}

output "codebuild_training_project" {
  description = "CodeBuild project name for GR00T training container"
  value       = aws_codebuild_project.groot_training.name
}

output "codebuild_inference_project" {
  description = "CodeBuild project name for GR00T inference container"
  value       = aws_codebuild_project.groot_inference.name
}
