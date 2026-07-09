# SPDX-License-Identifier: Apache-2.0

# AWS IAM Identity Center — Customer-managed OAuth 2.0 application for OSMO
#
# Creates the application, assigns scopes, configures the authorization_code
# grant with redirect URIs, and sets up the IAM authentication method.
# The client_secret must still be generated manually in the Identity Center
# console after `terraform apply`.

#------------------------------------------------------------------------------
# Look up the Identity Center instance
#------------------------------------------------------------------------------

data "aws_ssoadmin_instances" "this" {
  count = var.deploy_identity_center ? 1 : 0
}

locals {
  idc_instance_arn = var.deploy_identity_center ? tolist(data.aws_ssoadmin_instances.this[0].arns)[0] : ""
  idc_region       = var.deploy_identity_center ? (var.idc_region != "" ? var.idc_region : var.aws_region) : ""

  # Extract instance ID from the ARN: arn:aws:sso:::instance/ssoins-XXXX
  idc_instance_id_from_arn = var.deploy_identity_center ? regex("instance/(ssoins-[a-z0-9]+)", local.idc_instance_arn)[0] : ""
}

#------------------------------------------------------------------------------
# Customer-managed OAuth 2.0 Application
#------------------------------------------------------------------------------

resource "aws_ssoadmin_application" "osmo" {
  count = var.deploy_identity_center ? 1 : 0

  name                     = "${var.name_prefix}-osmo"
  instance_arn             = local.idc_instance_arn
  application_provider_arn = "arn:aws:sso::aws:applicationProvider/custom"
  description              = "OSMO on AWS — OAuth 2.0 OIDC provider"
  status                   = "ENABLED"

  portal_options {
    visibility = "ENABLED"
    sign_in_options {
      origin          = "APPLICATION"
      application_url = "https://${var.osmo_hostname}"
    }
  }

  tags = merge(var.common_tags, {
    Name = "${var.name_prefix}-idc-osmo"
  })
}

#------------------------------------------------------------------------------
# Access scopes — openid, email, profile
#------------------------------------------------------------------------------

resource "aws_ssoadmin_application_access_scope" "openid" {
  count = var.deploy_identity_center ? 1 : 0

  application_arn = aws_ssoadmin_application.osmo[0].application_arn
  scope           = "openid"
}

resource "aws_ssoadmin_application_access_scope" "email" {
  count = var.deploy_identity_center ? 1 : 0

  application_arn = aws_ssoadmin_application.osmo[0].application_arn
  scope           = "email"
}

resource "aws_ssoadmin_application_access_scope" "profile" {
  count = var.deploy_identity_center ? 1 : 0

  application_arn = aws_ssoadmin_application.osmo[0].application_arn
  scope           = "profile"
}

#------------------------------------------------------------------------------
# Authorization Code Grant + Redirect URIs (via AWS CLI — no Terraform resource yet)
#------------------------------------------------------------------------------

resource "terraform_data" "idc_grant" {
  count = var.deploy_identity_center ? 1 : 0

  triggers_replace = [
    aws_ssoadmin_application.osmo[0].application_arn,
    var.osmo_hostname,
  ]

  provisioner "local-exec" {
    command = <<-EOT
      aws sso-admin put-application-grant \
        --application-arn '${aws_ssoadmin_application.osmo[0].application_arn}' \
        --grant-type 'urn:ietf:params:oauth:grant-type:authorization_code' \
        --grant '{"AuthorizationCode":{"RedirectUris":["https://${var.osmo_hostname}/oauth2/callback"]}}' \
        --region '${local.idc_region}'
    EOT
  }
}

#------------------------------------------------------------------------------
# IAM Authentication Method (via AWS CLI — no Terraform resource yet)
# This enables the application for OIDC; the client_secret is created
# separately in the Identity Center console.
#------------------------------------------------------------------------------

resource "terraform_data" "idc_auth_method" {
  count = var.deploy_identity_center ? 1 : 0

  triggers_replace = [
    aws_ssoadmin_application.osmo[0].application_arn,
  ]

  provisioner "local-exec" {
    command = <<-EOT
      aws sso-admin put-application-authentication-method \
        --application-arn '${aws_ssoadmin_application.osmo[0].application_arn}' \
        --authentication-method-type IAM \
        --authentication-method '{"Iam":{"ActorPolicy":{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":"*"},"Action":"sso-oauth:CreateTokenWithIAM"}]}}}' \
        --region '${local.idc_region}'
    EOT
  }

  depends_on = [terraform_data.idc_grant]
}
