# =============================================================================
# VPC: Read shared VPC from Foundation (via SSM)
# The Foundation stack creates the VPC; this component just looks it up.
# =============================================================================

data "aws_ssm_parameter" "vpc_id" {
  name = "/${var.project_name}/vpc-id"
}

data "aws_ssm_parameter" "private_subnet_ids" {
  name = "/${var.project_name}/private-subnet-ids"
}

locals {
  foundation_vpc_id     = data.aws_ssm_parameter.vpc_id.value
  foundation_subnet_ids = split(",", data.aws_ssm_parameter.private_subnet_ids.value)
}
