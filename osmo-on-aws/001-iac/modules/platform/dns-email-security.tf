# SPDX-License-Identifier: Apache-2.0

#------------------------------------------------------------------------------
# Email anti-spoofing (SPF + DMARC)
#
# OSMO does not send email, so its domains should publish records that tell the
# world "no host is authorized to send mail as these names — reject anything that
# claims to be." This is the standard "no-mail" hardening for domains that never
# originate email, and it stops attackers from spoofing mail From: these names.
# See SPF (RFC 7208) and DMARC (RFC 7489). The records per protected name are:
#
#   <domain>          TXT  "v=spf1 -all"
#   _dmarc.<domain>   TXT  "v=DMARC1; p=reject; rua=mailto:<addr>; ruf=mailto:<addr>"
#
# DMARC p=reject at a name's _dmarc record is the control that actually stops
# spoofing: a receiver rejects any message whose visible From: is that domain and
# which fails authentication. It is published for EVERY protected name (apex +
# both OSMO hostnames). Because a _dmarc record is published directly for each
# name, coverage does not depend on organizational-domain / Public Suffix List
# resolution, and _dmarc.* names never collide with anything else in the zone.
#
# SPF (a TXT at the *bare* name) is published for the zone apex only. Publishing
# SPF on the two OSMO ingress hostnames is a deliberate NON-goal: DMARC p=reject
# below already rejects any spoofed mail whose visible From: is one of those
# names, so a per-host SPF record would add nothing. (Those hostnames are also
# ingress targets managed by external-dns, which keeps its own ownership TXT at
# the bare hostname.) For the hostnames we therefore rely on DMARC alone; the
# apex keeps SPF "-all" as a complementary no-mail marker because the apex is
# never an ingress target and cannot collide.
#
# allow_overwrite is left false so a pre-existing conflicting record makes
# `terraform apply` fail loudly instead of clobbering it.
#
# WARNING — before enabling any outbound mail from these names: this "no-mail"
# posture is correct only while nothing sends email From: them. As deployed the
# OSMO backend has no mailer; Keycloak's realm allows password reset but sets no
# SMTP server/From, so it cannot send; and Cognito uses its default
# no-reply@verificationemail.com sender (not these domains). If you later enable
# a real sender From: one of these names — e.g. a Keycloak SMTP server, or Cognito
# switched to a DEVELOPER/SES sender at osmo-aws-auth — that mail would be REJECTED
# under p=reject. First publish matching SPF + DKIM for that name and relax its
# DMARC to p=none (monitor) or p=quarantine.
#------------------------------------------------------------------------------

locals {
  # DMARC is safe on every protected name (external-dns never manages _dmarc.*).
  email_dmarc_domains = var.enable_email_spoofing_protection ? toset([
    var.route53_zone_name,
    var.osmo_hostname,
    var.osmo_auth_hostname,
  ]) : toset([])

  # SPF at the bare name, apex only. The zone apex is never an ingress target, so
  # its SPF "-all" is always safe. We deliberately publish no per-host SPF for the
  # OSMO hostnames — DMARC p=reject above already covers them.
  email_spf_domains = var.enable_email_spoofing_protection ? toset([
    var.route53_zone_name,
  ]) : toset([])

  # DMARC policy for a non-sending domain. Aggregate/forensic report addresses
  # (rua/ruf) are optional — p=reject enforces on its own — so they are included
  # only when dmarc_report_address is set.
  dmarc_record_value = var.dmarc_report_address != "" ? "v=DMARC1; p=reject; rua=mailto:${var.dmarc_report_address}; ruf=mailto:${var.dmarc_report_address}" : "v=DMARC1; p=reject"
}

# SPF: assert that no server is authorized to send mail as this domain.
resource "aws_route53_record" "spf" {
  for_each = local.email_spf_domains

  zone_id = var.route53_zone_id
  name    = each.value
  type    = "TXT"
  ttl     = 300
  records = ["v=spf1 -all"]
}

# DMARC: reject mail that fails authentication and send reports to the aggregator.
resource "aws_route53_record" "dmarc" {
  for_each = local.email_dmarc_domains

  zone_id = var.route53_zone_id
  name    = "_dmarc.${each.value}"
  type    = "TXT"
  ttl     = 300
  records = [local.dmarc_record_value]
}
