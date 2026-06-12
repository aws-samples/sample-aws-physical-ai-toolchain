import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as eks from 'aws-cdk-lib/aws-eks';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import { KubectlV31Layer } from '@aws-cdk/lambda-layer-kubectl-v31';
import { Construct } from 'constructs';

export interface EksClusterStackProps extends cdk.StackProps {
  environment: string;
  vpc: ec2.Vpc;
  eksSecurityGroup: ec2.SecurityGroup;
  eksConfig: {
    clusterName: string;
    version: string;
    controlPlaneNodes: {
      instanceType: string;
      minSize: number;
      maxSize: number;
      desiredSize: number;
    };
    gpuNodes: {
      instanceType: string;
      minSize: number;
      maxSize: number;
      desiredSize: number;
      diskSize: number;
    };
  };
  // S3 buckets for node access
  assetsBucket: s3.Bucket;
  checkpointsBucket: s3.Bucket;
  modelsBucket: s3.Bucket;
  telemetryBucket: s3.Bucket;
  // ECR repos for image pulls
  isaacSimRepo: ecr.Repository;
  isaacLabRepo: ecr.Repository;
  inferenceRepo: ecr.Repository;
}

/**
 * EKS Cluster stack: Single cluster with multiple node pools.
 *
 * Node Pools:
 * - Control pool (m6i): OSMO control plane, API server, scheduler
 * - GPU pool (g5/P5e): Isaac Sim simulation + Isaac Lab training
 *
 * The GPU pool scales from 0 — OSMO triggers scale-up when workflows are submitted.
 * This avoids burning GPU costs when idle.
 *
 * NVIDIA GPU support:
 * - NVIDIA device plugin DaemonSet (auto-installed via EKS GPU AMI)
 * - Nodes use EKS-optimized GPU AMI with pre-installed CUDA drivers
 */
export class EksClusterStack extends cdk.Stack {
  public readonly cluster: eks.Cluster;

  constructor(scope: Construct, id: string, props: EksClusterStackProps) {
    super(scope, id, props);

    const { eksConfig, vpc, eksSecurityGroup } = props;

    // IAM role for EKS cluster
    const clusterRole = new iam.Role(this, 'ClusterRole', {
      assumedBy: new iam.ServicePrincipal('eks.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEKSClusterPolicy'),
      ],
    });

    // Create EKS cluster
    this.cluster = new eks.Cluster(this, 'PhysicalAiCluster', {
      clusterName: eksConfig.clusterName,
      version: eks.KubernetesVersion.of(eksConfig.version),
      vpc,
      vpcSubnets: [{ subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS }],
      securityGroup: eksSecurityGroup,
      defaultCapacity: 0, // We manage node groups explicitly
      endpointAccess: eks.EndpointAccess.PUBLIC_AND_PRIVATE,
      role: clusterRole,
      kubectlLayer: new KubectlV31Layer(this, 'KubectlLayer'),
      mastersRole: new iam.Role(this, 'ClusterAdminRole', {
        assumedBy: new iam.AccountRootPrincipal(),
        roleName: `physical-ai-${props.environment}-cluster-admin`,
      }),
    });

    // Grant the deploying IAM user cluster admin access
    this.cluster.awsAuth.addUserMapping(
      iam.User.fromUserName(this, 'AdminUser', 'SteveAdmin'),
      { groups: ['system:masters'] }
    );

    // --- Control Plane Node Group ---
    // Runs OSMO services, monitoring, and lightweight workloads
    const controlNodeGroup = this.cluster.addNodegroupCapacity('ControlNodes', {
      nodegroupName: 'control-plane',
      instanceTypes: [new ec2.InstanceType(eksConfig.controlPlaneNodes.instanceType)],
      minSize: eksConfig.controlPlaneNodes.minSize,
      maxSize: eksConfig.controlPlaneNodes.maxSize,
      desiredSize: eksConfig.controlPlaneNodes.desiredSize,
      diskSize: 100,
      labels: {
        'node-role': 'control',
        'workload-type': 'osmo',
      },
      taints: [], // No taints — general workloads can schedule here
    });

    // --- GPU Node Group ---
    // Runs Isaac Sim (simulation) and Isaac Lab (training)
    // Scales from 0 to save costs when no workflows are running
    const gpuNodeGroup = this.cluster.addNodegroupCapacity('GpuNodes', {
      nodegroupName: 'gpu-workers',
      instanceTypes: [new ec2.InstanceType(eksConfig.gpuNodes.instanceType)],
      minSize: eksConfig.gpuNodes.minSize,
      maxSize: eksConfig.gpuNodes.maxSize,
      desiredSize: eksConfig.gpuNodes.desiredSize,
      diskSize: eksConfig.gpuNodes.diskSize,
      amiType: eks.NodegroupAmiType.AL2_X86_64_GPU, // EKS GPU AMI with NVIDIA drivers
      labels: {
        'node-role': 'gpu-worker',
        'workload-type': 'simulation-training',
        'nvidia.com/gpu': 'true',
      },
      taints: [
        {
          key: 'nvidia.com/gpu',
          value: 'true',
          effect: eks.TaintEffect.NO_SCHEDULE,
        },
      ],
    });

    // --- IAM: Grant GPU nodes access to S3 and ECR ---
    const gpuNodeRole = gpuNodeGroup.role;

    // S3 access for training data, checkpoints, models
    props.assetsBucket.grantRead(gpuNodeRole);
    props.checkpointsBucket.grantReadWrite(gpuNodeRole);
    props.modelsBucket.grantReadWrite(gpuNodeRole);
    props.telemetryBucket.grantWrite(gpuNodeRole);

    // ECR access for pulling Isaac Sim / Lab containers
    props.isaacSimRepo.grantPull(gpuNodeRole);
    props.isaacLabRepo.grantPull(gpuNodeRole);
    props.inferenceRepo.grantPull(gpuNodeRole);

    // Also grant control nodes ECR access (for OSMO containers)
    const controlNodeRole = controlNodeGroup.role;
    props.isaacSimRepo.grantPull(controlNodeRole);
    props.isaacLabRepo.grantPull(controlNodeRole);
    props.inferenceRepo.grantPull(controlNodeRole);

    // --- NVIDIA Device Plugin ---
    // NOTE: The EKS GPU AMI (AL2_X86_64_GPU) already includes the NVIDIA device plugin.
    // No need to install it separately via Helm. The GPU AMI handles:
    // - NVIDIA drivers
    // - NVIDIA container toolkit
    // - NVIDIA device plugin DaemonSet
    // If using a non-GPU AMI, uncomment the Helm chart below.

    // --- Install Cluster Autoscaler ---
    // Enables GPU nodes to scale from 0 when OSMO submits workflows
    this.cluster.addHelmChart('ClusterAutoscaler', {
      chart: 'cluster-autoscaler',
      repository: 'https://kubernetes.github.io/autoscaler',
      namespace: 'kube-system',
      values: {
        autoDiscovery: {
          clusterName: eksConfig.clusterName,
        },
        awsRegion: this.region,
        extraArgs: {
          'scale-down-delay-after-add': '10m',
          'scale-down-unneeded-time': '10m',
          'skip-nodes-with-local-storage': 'false',
        },
      },
    });

    // --- Outputs ---
    new cdk.CfnOutput(this, 'ClusterName', { value: this.cluster.clusterName });
    new cdk.CfnOutput(this, 'ClusterEndpoint', { value: this.cluster.clusterEndpoint });
    new cdk.CfnOutput(this, 'ClusterArn', { value: this.cluster.clusterArn });
  }
}
