# SPDX-License-Identifier: Apache-2.0

# AWS Cognito User Pool, App Clients, Custom Domain, Groups, and initial admin user

#------------------------------------------------------------------------------
# User Pool
#------------------------------------------------------------------------------

resource "aws_cognito_user_pool" "osmo" {
  count = var.deploy_cognito ? 1 : 0

  name = "${var.name_prefix}-osmo"

  username_attributes      = ["email"]
  auto_verified_attributes = ["email"]

  username_configuration {
    case_sensitive = false
  }

  password_policy {
    minimum_length                   = 8
    require_lowercase                = true
    require_numbers                  = true
    require_symbols                  = false
    require_uppercase                = true
    temporary_password_validity_days = 7
  }

  schema {
    name                     = "email"
    attribute_data_type      = "String"
    required                 = true
    mutable                  = true
    developer_only_attribute = false

    string_attribute_constraints {
      min_length = 1
      max_length = 256
    }
  }

  schema {
    name                     = "preferred_username"
    attribute_data_type      = "String"
    required                 = false
    mutable                  = true
    developer_only_attribute = false

    string_attribute_constraints {
      min_length = 1
      max_length = 256
    }
  }

  admin_create_user_config {
    allow_admin_create_user_only = false
  }

  account_recovery_setting {
    recovery_mechanism {
      name     = "verified_email"
      priority = 1
    }
  }

  tags = merge(var.common_tags, {
    Name = "${var.name_prefix}-cognito-pool"
  })
}

#------------------------------------------------------------------------------
# Custom Domain (uses ACM cert from us-east-1 for CloudFront)
# Cognito requires the parent domain of the custom domain to have a resolvable A
# record. When cognito_parent_domain_placeholder_ip is set, we create that A
# record first so CreateUserPoolDomain succeeds.
#------------------------------------------------------------------------------

data "aws_route53_zone" "cognito" {
  count = (var.deploy_cognito && var.cognito_parent_domain_placeholder_ip != "") ? 1 : 0

  zone_id = var.route53_zone_id
}

locals {
  # Parent domain of osmo_auth_hostname (e.g. osmo-aws-auth.example.com -> example.com)
  cognito_parent_domain = var.deploy_cognito ? join(".", slice(split(".", var.osmo_auth_hostname), 1, length(split(".", var.osmo_auth_hostname)))) : ""
}

resource "aws_route53_record" "cognito_parent_domain" {
  count = var.deploy_cognito && var.cognito_parent_domain_placeholder_ip != "" ? 1 : 0

  zone_id = var.route53_zone_id
  name    = local.cognito_parent_domain == data.aws_route53_zone.cognito[0].name ? "" : trim_suffix(local.cognito_parent_domain, ".${data.aws_route53_zone.cognito[0].name}")
  type    = "A"
  ttl     = 60
  records = [var.cognito_parent_domain_placeholder_ip]
}

resource "aws_cognito_user_pool_domain" "osmo" {
  count = var.deploy_cognito ? 1 : 0

  domain          = var.osmo_auth_hostname
  user_pool_id    = aws_cognito_user_pool.osmo[0].id
  certificate_arn = aws_acm_certificate_validation.osmo_auth[0].certificate_arn

  depends_on = [aws_route53_record.cognito_parent_domain]
}

#------------------------------------------------------------------------------
# App Client — Browser (Authorization Code Grant with PKCE + secret)
#------------------------------------------------------------------------------

resource "aws_cognito_user_pool_client" "browser" {
  count = var.deploy_cognito ? 1 : 0

  name         = "${var.name_prefix}-browser"
  user_pool_id = aws_cognito_user_pool.osmo[0].id

  generate_secret = true

  allowed_oauth_flows_user_pool_client = true
  allowed_oauth_flows                  = ["code"]
  allowed_oauth_scopes                 = ["openid", "email", "profile"]

  supported_identity_providers = ["COGNITO"]

  callback_urls = [
    "https://${var.osmo_hostname}/oauth2/callback"
  ]

  logout_urls = [
    "https://${var.osmo_hostname}/oauth2/sign_out"
  ]

  explicit_auth_flows = [
    "ALLOW_REFRESH_TOKEN_AUTH",
  ]

  token_validity_units {
    access_token  = "hours"
    id_token      = "hours"
    refresh_token = "days"
  }

  access_token_validity  = 1
  id_token_validity      = 1
  refresh_token_validity = 7
}

#------------------------------------------------------------------------------
# App Client — CLI (Password-based auth, no secret)
#------------------------------------------------------------------------------

resource "aws_cognito_user_pool_client" "cli" {
  count = var.deploy_cognito ? 1 : 0

  name         = "${var.name_prefix}-cli"
  user_pool_id = aws_cognito_user_pool.osmo[0].id

  generate_secret = false

  allowed_oauth_flows_user_pool_client = false

  explicit_auth_flows = [
    "ALLOW_USER_PASSWORD_AUTH",
    "ALLOW_REFRESH_TOKEN_AUTH",
  ]

  token_validity_units {
    access_token  = "hours"
    id_token      = "hours"
    refresh_token = "days"
  }

  access_token_validity  = 1
  id_token_validity      = 1
  refresh_token_validity = 7
}

#------------------------------------------------------------------------------
# User Groups
#------------------------------------------------------------------------------

resource "aws_cognito_user_group" "admin" {
  count = var.deploy_cognito ? 1 : 0

  name         = "osmo-admin"
  user_pool_id = aws_cognito_user_pool.osmo[0].id
  description  = "OSMO administrators with full access"
  precedence   = 1
}

resource "aws_cognito_user_group" "user" {
  count = var.deploy_cognito ? 1 : 0

  name         = "osmo-user"
  user_pool_id = aws_cognito_user_pool.osmo[0].id
  description  = "Standard OSMO users"
  precedence   = 10
}

#------------------------------------------------------------------------------
# Initial Admin User
#------------------------------------------------------------------------------

resource "aws_cognito_user" "admin" {
  count = var.deploy_cognito && var.cognito_admin_email != "" ? 1 : 0

  user_pool_id = aws_cognito_user_pool.osmo[0].id
  username     = var.cognito_admin_email

  attributes = {
    email              = var.cognito_admin_email
    email_verified     = "true"
    preferred_username = var.cognito_admin_username
  }

  temporary_password = var.cognito_admin_temp_password

  lifecycle {
    ignore_changes = [temporary_password]
  }
}

resource "aws_cognito_user_in_group" "admin_group" {
  count = var.deploy_cognito && var.cognito_admin_email != "" ? 1 : 0

  user_pool_id = aws_cognito_user_pool.osmo[0].id
  group_name   = aws_cognito_user_group.admin[0].name
  username     = aws_cognito_user.admin[0].username
}

#------------------------------------------------------------------------------
# Route53 Alias for Cognito Custom Domain
#------------------------------------------------------------------------------

resource "aws_route53_record" "cognito_custom_domain" {
  count = var.deploy_cognito ? 1 : 0

  zone_id         = var.route53_zone_id
  name            = var.osmo_auth_hostname
  type            = "A"
  allow_overwrite = true

  alias {
    name                   = aws_cognito_user_pool_domain.osmo[0].cloudfront_distribution
    zone_id                = aws_cognito_user_pool_domain.osmo[0].cloudfront_distribution_zone_id
    evaluate_target_health = false
  }
}
