# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TFLint configuration for OSMO on AWS
# Documentation: https://github.com/terraform-linters/tflint

config {
  # Enable module inspection
  call_module_type = "local"

  # Force to return non-zero exit code
  force = false

  # Disable specific rules
  disabled_by_default = false
}

# AWS Plugin
plugin "aws" {
  enabled = true
  version = "0.32.0"
  source  = "github.com/terraform-linters/tflint-ruleset-aws"
}

# Terraform language rules
plugin "terraform" {
  enabled = true
  preset  = "recommended"
}

# Naming convention rules
rule "terraform_naming_convention" {
  enabled = true

  # Variables should be snake_case
  variable {
    format = "snake_case"
  }

  # Locals should be snake_case
  locals {
    format = "snake_case"
  }

  # Outputs should be snake_case
  output {
    format = "snake_case"
  }

  # Resources should be snake_case
  resource {
    format = "snake_case"
  }

  # Data sources should be snake_case
  data {
    format = "snake_case"
  }

  # Modules should be snake_case
  module {
    format = "snake_case"
  }
}

# Require descriptions for variables
rule "terraform_documented_variables" {
  enabled = true
}

# Require descriptions for outputs
rule "terraform_documented_outputs" {
  enabled = true
}

# Require type declarations for variables
rule "terraform_typed_variables" {
  enabled = true
}

# Standard module structure
rule "terraform_standard_module_structure" {
  enabled = true
}

# Unused declarations
rule "terraform_unused_declarations" {
  enabled = true
}

# Deprecated syntax
rule "terraform_deprecated_interpolation" {
  enabled = true
}

# AWS-specific rules
rule "aws_instance_invalid_type" {
  enabled = true
}

rule "aws_db_instance_invalid_type" {
  enabled = true
}

rule "aws_elasticache_cluster_invalid_type" {
  enabled = true
}

# Suppress notice about using default Redis parameter group.
# A custom parameter group can be added when tuning is needed.
rule "aws_elasticache_replication_group_default_parameter_group" {
  enabled = false
}
