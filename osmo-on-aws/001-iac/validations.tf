# SPDX-License-Identifier: Apache-2.0

# Guardrail for derived resource-name length.
#
# EKS managed node groups name their IAM role "<name_prefix>-<system|gpu>-eks-node-group-",
# and AWS limits an IAM role name_prefix to 38 characters. The longest suffix,
# "-system-eks-node-group-", is 23 chars, so local.name_prefix must be <= 15.
# Fail fast here with an actionable message instead of the cryptic module-level
# "expected length of name_prefix to be in the range (1 - 38)" error.
resource "terraform_data" "name_prefix_length_guard" {
  lifecycle {
    precondition {
      condition     = length(local.name_prefix) <= 15
      error_message = "name_prefix \"${local.name_prefix}\" is ${length(local.name_prefix)} characters, but must be <= 15. It is built as cluster_name-environment-resource_suffix and feeds EKS node-group IAM role names, which AWS caps at 38 characters. Shorten cluster_name and/or resource_suffix (environment is fixed to dev/staging/prod)."
    }
  }
}
