terraform {
  required_version = ">= 1.9"

  required_providers {
    aws = {
      source = "hashicorp/aws"
      # S8: pinned to the MAJOR version this component is actually deployed with, and the
      # dependency lock file is committed rather than ignored.
      #
      # The constraint was ">= 5.60" while the lock recorded 6.64.0, so a fresh clone --
      # which the README promises works from scratch -- resolved across the 5.x/6.x major
      # boundary to whatever was latest. The pessimistic constraint accepts 6.64.x patch
      # releases and refuses 7.x, and the committed lock makes every clone select the same
      # provider build. Committing the lock is HashiCorp's own recommendation for exactly
      # this reason; ignoring it defeated the pinning it exists to provide.
      version = "~> 6.64"
    }
  }
}

provider "aws" {
  region = var.aws_region
}
