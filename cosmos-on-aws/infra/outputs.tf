output "cosmos_transfer_ecr_uri" {
  description = "ECR URI for the Cosmos Transfer 2.5 container"
  value       = aws_ecr_repository.cosmos_transfer.repository_url
}

output "cosmos3_ecr_uri" {
  description = "ECR URI for the Cosmos 3 container"
  value       = aws_ecr_repository.cosmos3.repository_url
}

output "codebuild_transfer_project" {
  description = "CodeBuild project for Cosmos Transfer container"
  value       = aws_codebuild_project.cosmos_transfer.name
}

output "codebuild_cosmos3_project" {
  description = "CodeBuild project for Cosmos 3 container"
  value       = aws_codebuild_project.cosmos3.name
}
