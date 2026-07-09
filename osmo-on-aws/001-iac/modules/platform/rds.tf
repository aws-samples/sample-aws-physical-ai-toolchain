# SPDX-License-Identifier: Apache-2.0

# Generate random password for RDS
resource "random_password" "rds_password" {
  count   = var.deploy_postgresql ? 1 : 0
  length  = 32
  special = false # Avoid special chars that may cause issues with connection strings
}

# RDS PostgreSQL instance
module "rds" {
  source  = "terraform-aws-modules/rds/aws"
  version = "~> 6.0"

  count = var.deploy_postgresql ? 1 : 0

  identifier = "${var.name_prefix}-postgresql"

  # Engine configuration
  engine               = "postgres"
  engine_version       = var.rds_engine_version
  family               = "postgres${split(".", var.rds_engine_version)[0]}"
  major_engine_version = split(".", var.rds_engine_version)[0]
  instance_class       = var.rds_instance_class

  # AWS auto-upgrades minor versions during maintenance windows, which makes the
  # live engine_version drift ahead of this pinned value. Since RDS has no
  # downgrade path, a later apply would otherwise fail. Disable it so the pinned
  # version stays authoritative; bump rds_engine_version explicitly to upgrade.
  auto_minor_version_upgrade = false

  # Storage configuration
  allocated_storage     = var.rds_allocated_storage
  max_allocated_storage = var.rds_max_allocated_storage
  storage_type          = "gp3"
  storage_encrypted     = true
  kms_key_id            = aws_kms_key.osmo.arn

  # Database configuration
  db_name  = var.rds_db_name
  username = var.rds_username
  password = random_password.rds_password[0].result
  port     = 5432

  # Disable AWS Secrets Manager managed password (we manage it ourselves)
  manage_master_user_password = false

  # Network configuration
  multi_az               = var.rds_multi_az
  publicly_accessible    = false
  vpc_security_group_ids = [aws_security_group.rds[0].id]
  db_subnet_group_name   = module.vpc.database_subnet_group_name

  # Backup configuration
  backup_retention_period = var.rds_backup_retention_period
  backup_window           = "03:00-04:00"
  maintenance_window      = "Sun:04:00-Sun:05:00"
  copy_tags_to_snapshot   = true
  skip_final_snapshot     = var.environment != "prod"
  deletion_protection     = var.rds_deletion_protection

  # Monitoring
  monitoring_interval             = 60
  monitoring_role_name            = "${var.name_prefix}-rds-monitoring-role"
  create_monitoring_role          = true
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]

  # Performance Insights
  performance_insights_enabled          = true
  performance_insights_retention_period = 7
  performance_insights_kms_key_id       = aws_kms_key.osmo.arn

  # Parameter group
  parameters = [
    {
      name  = "log_connections"
      value = "1"
    },
    {
      name  = "log_disconnections"
      value = "1"
    },
    {
      name  = "log_statement"
      value = "ddl"
    }
  ]

  tags = var.common_tags
}

# Security group for RDS
resource "aws_security_group" "rds" {
  count = var.deploy_postgresql ? 1 : 0

  name        = "${var.name_prefix}-rds-sg"
  description = "Security group for RDS PostgreSQL"
  vpc_id      = module.vpc.vpc_id

  tags = merge(var.common_tags, {
    Name = "${var.name_prefix}-rds-sg"
  })

  lifecycle {
    create_before_destroy = true
  }
}

# Allow ingress from VPC CIDR (for initial setup and debugging)
# Disabled in production when restrict_rds_to_eks_only is true
resource "aws_security_group_rule" "rds_from_vpc" {
  count = var.deploy_postgresql && !var.restrict_rds_to_eks_only ? 1 : 0

  type              = "ingress"
  from_port         = 5432
  to_port           = 5432
  protocol          = "tcp"
  cidr_blocks       = [var.vpc_cidr]
  security_group_id = aws_security_group.rds[0].id
  description       = "PostgreSQL from VPC (dev/debug only)"
}

# Egress rule
resource "aws_security_group_rule" "rds_egress" {
  count = var.deploy_postgresql ? 1 : 0

  type              = "egress"
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
  security_group_id = aws_security_group.rds[0].id
  description       = "All outbound traffic"
}

# Store RDS password in Secrets Manager
resource "aws_secretsmanager_secret" "rds_password" {
  count = var.deploy_postgresql ? 1 : 0

  name                    = "${var.name_prefix}/rds/password"
  description             = "RDS PostgreSQL master password"
  recovery_window_in_days = var.secrets_recovery_window_days
  kms_key_id              = aws_kms_key.osmo.arn

  tags = var.common_tags
}

resource "aws_secretsmanager_secret_version" "rds_password" {
  count = var.deploy_postgresql ? 1 : 0

  secret_id = aws_secretsmanager_secret.rds_password[0].id
  secret_string = jsonencode({
    username = var.rds_username
    password = random_password.rds_password[0].result
    host     = module.rds[0].db_instance_endpoint
    port     = 5432
    database = var.rds_db_name
    engine   = "postgres"
  })
}
