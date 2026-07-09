# SPDX-License-Identifier: Apache-2.0

# ACM Certificates and Route53 DNS validation for OSMO service and auth endpoints

#------------------------------------------------------------------------------
# ACM Certificate - OSMO Service
#------------------------------------------------------------------------------

resource "aws_acm_certificate" "osmo" {
  domain_name       = var.osmo_hostname
  validation_method = "DNS"

  tags = merge(var.common_tags, {
    Name = "${var.name_prefix}-osmo-cert"
  })

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_route53_record" "osmo_cert_validation" {
  for_each = {
    for dvo in aws_acm_certificate.osmo.domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  }

  zone_id         = var.route53_zone_id
  name            = each.value.name
  type            = each.value.type
  ttl             = 60
  records         = [each.value.record]
  allow_overwrite = true
}

resource "aws_acm_certificate_validation" "osmo" {
  certificate_arn         = aws_acm_certificate.osmo.arn
  validation_record_fqdns = [for record in aws_route53_record.osmo_cert_validation : record.fqdn]
}

#------------------------------------------------------------------------------
# ACM Certificate - OSMO Auth (Cognito custom domain)
#
# Cognito custom domains are backed by CloudFront, which requires the ACM
# certificate in us-east-1.
#------------------------------------------------------------------------------

resource "aws_acm_certificate" "osmo_auth" {
  count    = var.deploy_cognito ? 1 : 0
  provider = aws.us_east_1

  domain_name       = var.osmo_auth_hostname
  validation_method = "DNS"

  tags = merge(var.common_tags, {
    Name = "${var.name_prefix}-osmo-auth-cert"
  })

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_route53_record" "osmo_auth_cert_validation" {
  for_each = var.deploy_cognito ? {
    for dvo in aws_acm_certificate.osmo_auth[0].domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  } : {}

  zone_id         = var.route53_zone_id
  name            = each.value.name
  type            = each.value.type
  ttl             = 60
  records         = [each.value.record]
  allow_overwrite = true
}

resource "aws_acm_certificate_validation" "osmo_auth" {
  count    = var.deploy_cognito ? 1 : 0
  provider = aws.us_east_1

  certificate_arn         = aws_acm_certificate.osmo_auth[0].arn
  validation_record_fqdns = [for record in aws_route53_record.osmo_auth_cert_validation : record.fqdn]
}

#------------------------------------------------------------------------------
# ACM Certificate - OSMO Auth (Keycloak)
#
# Keycloak runs behind the ALB in the deployment region, so the certificate
# is created in the same region (not us-east-1).  The DNS record pointing
# to the ALB is managed by the ALB Ingress Controller / external-dns.
#------------------------------------------------------------------------------

resource "aws_acm_certificate" "osmo_keycloak" {
  count = var.deploy_keycloak ? 1 : 0

  domain_name       = var.osmo_auth_hostname
  validation_method = "DNS"

  tags = merge(var.common_tags, {
    Name = "${var.name_prefix}-osmo-keycloak-cert"
  })

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_route53_record" "osmo_keycloak_cert_validation" {
  for_each = var.deploy_keycloak ? {
    for dvo in aws_acm_certificate.osmo_keycloak[0].domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  } : {}

  zone_id         = var.route53_zone_id
  name            = each.value.name
  type            = each.value.type
  ttl             = 60
  records         = [each.value.record]
  allow_overwrite = true
}

resource "aws_acm_certificate_validation" "osmo_keycloak" {
  count = var.deploy_keycloak ? 1 : 0

  certificate_arn         = aws_acm_certificate.osmo_keycloak[0].arn
  validation_record_fqdns = [for record in aws_route53_record.osmo_keycloak_cert_validation : record.fqdn]
}
