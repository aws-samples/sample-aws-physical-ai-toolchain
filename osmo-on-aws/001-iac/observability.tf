# SPDX-License-Identifier: Apache-2.0

#------------------------------------------------------------------------------
# Observability
#   #1 Logs    — Fluent Bit IRSA role + CloudWatch log group (DaemonSet is
#                deployed by 002-setup/01-deploy-aws-prerequisites.sh).
#   #2 Metrics — Amazon Managed Prometheus (AMP) workspace + managed scraper
#                (agentless). Grafana is intentionally out of scope for now.
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# #1 Logs — Fluent Bit -> CloudWatch Logs
#------------------------------------------------------------------------------

# Log group for OSMO container logs shipped by Fluent Bit.
# NOTE: uses the AWS-managed CloudWatch encryption key. To use the platform CMK
# instead, the KMS key policy must grant logs.<region>.amazonaws.com access —
# omitted here to keep `terraform apply` self-contained.
resource "aws_cloudwatch_log_group" "osmo_logs" {
  count             = var.enable_cloudwatch_logging ? 1 : 0
  name              = "/aws/eks/${module.eks.cluster_name}/osmo-logs"
  retention_in_days = var.cloudwatch_logs_retention_days
  tags              = local.common_tags
}

# IAM policy for Fluent Bit to write to the log group above.
resource "aws_iam_policy" "fluentbit" {
  count       = var.enable_cloudwatch_logging ? 1 : 0
  name        = "${local.name_prefix}-fluentbit-logs"
  description = "CloudWatch Logs write access for the Fluent Bit log shipper"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CloudWatchLogsWrite"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:PutRetentionPolicy",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams"
        ]
        Resource = [
          aws_cloudwatch_log_group.osmo_logs[0].arn,
          "${aws_cloudwatch_log_group.osmo_logs[0].arn}:*"
        ]
      }
    ]
  })

  tags = local.common_tags
}

# IRSA role assumed by the aws-for-fluent-bit ServiceAccount.
module "fluentbit_irsa" {
  count   = var.enable_cloudwatch_logging ? 1 : 0
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "${local.name_prefix}-fluentbit-irsa"

  role_policy_arns = {
    logs = aws_iam_policy.fluentbit[0].arn
  }

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["amazon-cloudwatch:aws-for-fluent-bit"]
    }
  }

  tags = local.common_tags
}

#------------------------------------------------------------------------------
# #2 Metrics — Amazon Managed Prometheus (AMP) + managed scraper
#------------------------------------------------------------------------------

resource "aws_prometheus_workspace" "osmo" {
  count = var.enable_managed_prometheus ? 1 : 0
  alias = "${local.name_prefix}-osmo"
  tags  = local.common_tags
}

# Agentless managed scraper: AWS runs the collector in its own account, reaches
# pods/metrics endpoints via cross-account ENIs in the private subnets, and
# remote-writes to AMP. It authenticates to the cluster as `scraper.role_arn`,
# which is granted read access via the EKS access entry below + the
# `aps-collector` ClusterRole/Binding applied by 02-setup.
resource "aws_prometheus_scraper" "osmo" {
  count = var.enable_managed_prometheus ? 1 : 0

  source {
    eks {
      cluster_arn        = module.eks.cluster_arn
      subnet_ids         = module.platform.private_subnet_ids
      security_group_ids = [module.eks.node_security_group_id]
    }
  }

  destination {
    amp {
      workspace_arn = aws_prometheus_workspace.osmo[0].arn
    }
  }

  # Scrape kubelet/cAdvisor, kube-state-metrics, and any pod/endpoint annotated
  # with prometheus.io/scrape — which includes the GPU Operator's DCGM exporter.
  scrape_configuration = <<-EOT
    global:
      scrape_interval: 30s
    scrape_configs:
      - job_name: kubernetes-pods
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
            action: replace
            target_label: __metrics_path__
            regex: (.+)
          - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
            action: replace
            regex: ([^:]+)(?::\d+)?;(\d+)
            replacement: $1:$2
            target_label: __address__
          - source_labels: [__meta_kubernetes_namespace]
            action: replace
            target_label: namespace
          - source_labels: [__meta_kubernetes_pod_name]
            action: replace
            target_label: pod
      - job_name: kubernetes-nodes-cadvisor
        scheme: https
        tls_config:
          insecure_skip_verify: true
        authorization:
          credentials_file: /var/run/secrets/kubernetes.io/serviceaccount/token
        kubernetes_sd_configs:
          - role: node
        relabel_configs:
          - action: labelmap
            regex: __meta_kubernetes_node_label_(.+)
          - target_label: __metrics_path__
            replacement: /metrics/cadvisor
  EOT

  tags = local.common_tags
}

# NOTE: cluster access for the managed scraper is handled automatically by AMP.
# Because this cluster uses EKS access entries (authentication_mode API /
# API_AND_CONFIG_MAP), creating the scraper makes AMP auto-create an access
# entry for its service-linked role and associate the AWS-managed
# `AmazonPrometheusScraperPolicy` cluster access policy. We must NOT create the
# access entry ourselves — EKS rejects access entries whose principal is a
# service-linked role ("not allowed to modify access entries with a principalArn
# value of a Service Linked Role"). No manual ClusterRole/ClusterRoleBinding is
# needed either (that is only required for the legacy aws-auth ConfigMap path).
