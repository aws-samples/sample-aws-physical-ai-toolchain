output "cosmos_transfer_ecr_uri" {
  description = "ECR URI for the Cosmos Transfer 2.5 container"
  value       = data.aws_ssm_parameter.cosmos_transfer_ecr.value
}

output "cosmos3_ecr_uri" {
  description = "ECR URI for the Cosmos 3 container"
  value       = data.aws_ssm_parameter.cosmos3_ecr.value
}

output "codebuild_transfer_project" {
  description = "CodeBuild project for Cosmos Transfer container"
  value       = aws_codebuild_project.cosmos_transfer.name
}

output "codebuild_cosmos3_project" {
  description = "CodeBuild project for Cosmos 3 container"
  value       = aws_codebuild_project.cosmos3.name
}
