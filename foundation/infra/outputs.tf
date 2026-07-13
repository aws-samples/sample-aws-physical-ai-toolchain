output "datasets_bucket_name" {
  description = "S3 bucket for training datasets"
  value       = aws_s3_bucket.datasets.bucket
}

output "models_bucket_name" {
  description = "S3 bucket for trained models"
  value       = aws_s3_bucket.models.bucket
}

output "checkpoints_bucket_name" {
  description = "S3 bucket for training checkpoints"
  value       = aws_s3_bucket.checkpoints.bucket
}

output "sagemaker_role_arn" {
  description = "SageMaker execution role ARN"
  value       = aws_iam_role.sagemaker.arn
}

output "cosmos_instance_profile_name" {
  description = "Instance profile for Cosmos EC2 instances"
  value       = aws_iam_instance_profile.cosmos.name
}
