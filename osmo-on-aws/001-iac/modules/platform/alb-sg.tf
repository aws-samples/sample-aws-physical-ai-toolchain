# SPDX-License-Identifier: Apache-2.0

# ALB security group — restricts inbound HTTPS to specified CIDRs.
# Only created when alb_allowed_cidrs is non-empty; otherwise the AWS
# Load Balancer Controller manages its own security group automatically.

resource "aws_security_group" "alb" {
  count = length(var.alb_allowed_cidrs) > 0 ? 1 : 0

  name_prefix = "${var.name_prefix}-alb-"
  description = "OSMO ALB - restrict HTTPS to allowed CIDRs"
  vpc_id      = module.vpc.vpc_id

  tags = merge(var.common_tags, {
    Name = "${var.name_prefix}-alb"
  })

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_security_group_rule" "alb_ingress_https" {
  count = length(var.alb_allowed_cidrs) > 0 ? 1 : 0

  security_group_id = aws_security_group.alb[0].id
  type              = "ingress"
  from_port         = 443
  to_port           = 443
  protocol          = "tcp"
  cidr_blocks       = var.alb_allowed_cidrs
  description       = "HTTPS from allowed CIDRs"
}

resource "aws_security_group_rule" "alb_ingress_nat" {
  count = length(var.alb_allowed_cidrs) > 0 ? 1 : 0

  security_group_id = aws_security_group.alb[0].id
  type              = "ingress"
  from_port         = 443
  to_port           = 443
  protocol          = "tcp"
  cidr_blocks       = [for ip in module.vpc.nat_public_ips : "${ip}/32"]
  description       = "HTTPS from cluster pods via NAT Gateway"
}

resource "aws_security_group_rule" "alb_egress_all" {
  count = length(var.alb_allowed_cidrs) > 0 ? 1 : 0

  security_group_id = aws_security_group.alb[0].id
  type              = "egress"
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
  description       = "Allow all outbound (health checks, targets)"
}
