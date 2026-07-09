# SPDX-License-Identifier: Apache-2.0

locals {
  # System node group configuration
  system_node_group = {
    system = {
      name = "${var.name_prefix}-system"

      instance_types = var.system_node_instance_types
      capacity_type  = "ON_DEMAND"

      min_size     = var.system_node_min_size
      max_size     = var.system_node_max_size
      desired_size = var.system_node_desired_size

      # Use private subnets
      subnet_ids = var.private_subnets

      # Labels
      labels = {
        "node.kubernetes.io/purpose" = "system"
        "osmo/node-type"             = "system"
      }

      # Block device configuration
      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 100
            volume_type           = "gp3"
            iops                  = 3000
            throughput            = 125
            encrypted             = true
            kms_key_id            = var.kms_key_arn
            delete_on_termination = true
          }
        }
      }

      # Update configuration
      update_config = {
        max_unavailable_percentage = 33
      }

      tags = merge(var.common_tags, {
        "k8s.io/cluster-autoscaler/enabled"             = "true"
        "k8s.io/cluster-autoscaler/${var.cluster_name}" = "owned"
      })
    }
  }

  # Whether a custom AMI is being used (e.g. Canonical Ubuntu EKS)
  use_custom_gpu_ami = var.gpu_ami_id != ""

  # GPU node group configuration (conditional)
  gpu_node_group = var.deploy_gpu_nodes ? {
    gpu = {
      name = "${var.name_prefix}-gpu"

      instance_types = var.gpu_node_instance_types
      capacity_type  = var.gpu_node_capacity_type

      # Custom AMI (Ubuntu EKS) vs. managed AMI type
      ami_id   = local.use_custom_gpu_ami ? var.gpu_ami_id : null
      ami_type = local.use_custom_gpu_ami ? null : var.gpu_ami_type

      # When using a custom AMI the module must generate the
      # /etc/eks/bootstrap.sh call via launch-template user data.
      enable_bootstrap_user_data = local.use_custom_gpu_ami

      min_size     = var.gpu_node_min_size
      max_size     = var.gpu_node_max_size
      desired_size = var.gpu_node_desired_size

      # Use private subnets
      subnet_ids = var.private_subnets

      # Labels for GPU nodes
      labels = {
        "node.kubernetes.io/purpose" = "gpu"
        "osmo/node-type"             = "gpu"
        "nvidia.com/gpu.present"     = "true"
      }

      # Taints to prevent non-GPU workloads
      taints = var.gpu_taints

      # Block device configuration with larger disk for GPU workloads.
      # Must hold cached container images (Isaac Sim, Cosmos, GPU operator)
      # plus the kubelet's image-gc / eviction reserves.
      #
      # The launch-template device_name must match the AMI's root device or
      # EC2 attaches the EBS as an *additional* volume and the instance
      # boots from the AMI's default (small) root snapshot:
      #   - Amazon Linux 2 EKS-Optimized: /dev/xvda
      #   - Canonical Ubuntu EKS images:  /dev/sda1
      block_device_mappings = {
        root = {
          device_name = local.use_custom_gpu_ami ? "/dev/sda1" : "/dev/xvda"
          ebs = {
            volume_size           = var.gpu_node_volume_size
            volume_type           = "gp3"
            iops                  = 4000
            throughput            = 250
            encrypted             = true
            kms_key_id            = var.kms_key_arn
            delete_on_termination = true
          }
        }
      }

      # Update configuration
      update_config = {
        max_unavailable_percentage = 33
      }

      tags = merge(var.common_tags, {
        "k8s.io/cluster-autoscaler/enabled"                                    = "true"
        "k8s.io/cluster-autoscaler/${var.cluster_name}"                        = "owned"
        "k8s.io/cluster-autoscaler/node-template/label/nvidia.com/gpu.present" = "true"
        "k8s.io/cluster-autoscaler/node-template/taint/nvidia.com/gpu"         = "true:NoSchedule"
      })
    }
  } : {}
}
