output "isaac_lab_ecr_uri" {
  description = "ECR URI for the Isaac Lab training container"
  value       = aws_ecr_repository.isaac_lab.repository_url
}

output "codebuild_project" {
  description = "CodeBuild project name for Isaac Lab container"
  value       = aws_codebuild_project.isaac_lab.name
}
