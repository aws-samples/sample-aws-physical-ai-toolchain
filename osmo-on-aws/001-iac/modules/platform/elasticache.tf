# SPDX-License-Identifier: Apache-2.0

# Generate random auth token for Redis
resource "random_password" "redis_auth_token" {
  count   = var.deploy_redis ? 1 : 0
  length  = 64
  special = false # Redis auth tokens can only contain printable ASCII characters
}

# ElastiCache Redis replication group
resource "aws_elasticache_replication_group" "redis" {
  count = var.deploy_redis ? 1 : 0

  replication_group_id = "${var.name_prefix}-redis"
  description          = "OSMO Redis cluster for ${var.name_prefix}"

  # Engine configuration
  engine               = "redis"
  engine_version       = var.redis_engine_version
  node_type            = var.redis_node_type
  port                 = 6379
  parameter_group_name = "default.redis${split(".", var.redis_engine_version)[0]}"

  # Cluster configuration
  num_cache_clusters         = var.redis_num_cache_clusters
  automatic_failover_enabled = var.redis_automatic_failover_enabled
  multi_az_enabled           = var.redis_multi_az_enabled

  # Network configuration
  subnet_group_name  = module.vpc.elasticache_subnet_group_name
  security_group_ids = [aws_security_group.redis[0].id]

  # Security configuration
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = random_password.redis_auth_token[0].result
  kms_key_id                 = aws_kms_key.osmo.arn

  # Backup configuration
  snapshot_retention_limit = var.redis_snapshot_retention_limit
  snapshot_window          = "03:00-05:00"

  # Maintenance configuration
  maintenance_window         = "sun:05:00-sun:06:00"
  auto_minor_version_upgrade = true

  # Notification
  notification_topic_arn = null # Can be configured if SNS topic exists

  tags = merge(var.common_tags, {
    Name = "${var.name_prefix}-redis"
  })

  lifecycle {
    ignore_changes = [
      engine_version, # Avoid drift from minor version auto-upgrades
    ]
  }
}

# Security group for Redis
resource "aws_security_group" "redis" {
  count = var.deploy_redis ? 1 : 0

  name        = "${var.name_prefix}-redis-sg"
  description = "Security group for ElastiCache Redis"
  vpc_id      = module.vpc.vpc_id

  tags = merge(var.common_tags, {
    Name = "${var.name_prefix}-redis-sg"
  })

  lifecycle {
    create_before_destroy = true
  }
}

# Allow ingress from VPC CIDR (for initial setup and debugging)
# Disabled in production when restrict_redis_to_eks_only is true
resource "aws_security_group_rule" "redis_from_vpc" {
  count = var.deploy_redis && !var.restrict_redis_to_eks_only ? 1 : 0

  type              = "ingress"
  from_port         = 6379
  to_port           = 6379
  protocol          = "tcp"
  cidr_blocks       = [var.vpc_cidr]
  security_group_id = aws_security_group.redis[0].id
  description       = "Redis from VPC (dev/debug only)"
}

# Egress rule
resource "aws_security_group_rule" "redis_egress" {
  count = var.deploy_redis ? 1 : 0

  type              = "egress"
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
  security_group_id = aws_security_group.redis[0].id
  description       = "All outbound traffic"
}

# Store Redis auth token in Secrets Manager
resource "aws_secretsmanager_secret" "redis_auth_token" {
  count = var.deploy_redis ? 1 : 0

  name                    = "${var.name_prefix}/redis/auth-token"
  description             = "ElastiCache Redis auth token"
  recovery_window_in_days = var.secrets_recovery_window_days
  kms_key_id              = aws_kms_key.osmo.arn

  tags = var.common_tags
}

resource "aws_secretsmanager_secret_version" "redis_auth_token" {
  count = var.deploy_redis ? 1 : 0

  secret_id = aws_secretsmanager_secret.redis_auth_token[0].id
  secret_string = jsonencode({
    auth_token = random_password.redis_auth_token[0].result
    host       = aws_elasticache_replication_group.redis[0].primary_endpoint_address
    port       = 6379
    ssl        = true
  })
}
