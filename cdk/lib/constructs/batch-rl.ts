import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as efs from 'aws-cdk-lib/aws-efs';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as batch from 'aws-cdk-lib/aws-batch';
import { Construct } from 'constructs';

export interface BatchRlProps {
  readonly vpc: ec2.IVpc;
  readonly projectName: string;
  readonly environment: string;
  readonly instanceType?: string;
  readonly isaacLabRepo: ecr.IRepository;
  readonly efsFileSystem: efs.FileSystem;
  readonly maxvCpus?: number;
  readonly numNodes?: number;
}

/**
 * BatchRl — AWS Batch Multi-Node Parallel (MNP) RL training construct.
 *
 * Provisions:
 * - Managed EC2 compute environment (GPU instances, private subnet)
 * - Job queue
 * - MNP job definition (multi-node torchrun with EFS for shared checkpoints)
 *
 * The compute environment places ONE container per instance (by requesting
 * all GPUs + most of the vCPUs/memory) to ensure full NCCL topology visibility
 * per node. The job definition mounts EFS at /efs for persistent checkpoints
 * across nodes and runs.
 *
 * NOTE: Multi-node NCCL convergence is UNVALIDATED on hardware. This construct
 * wires the topology correctly, but end-to-end p2p GPU comms have not been
 * verified on g6 instances. See plan 02-batch-mnp.md for known risks.
 */
export class BatchRl extends Construct {
  public readonly queue: batch.JobQueue;
  public readonly jobDefinition: batch.MultiNodeJobDefinition;
  public readonly computeEnvironment: batch.ManagedEc2EcsComputeEnvironment;
  public readonly securityGroup: ec2.SecurityGroup;

  constructor(scope: Construct, id: string, props: BatchRlProps) {
    super(scope, id);

    const instanceType = props.instanceType || 'g6.12xlarge';
    const maxvCpus = props.maxvCpus || 96;
    const numNodes = props.numNodes || 2;

    // Security group for the compute instances
    this.securityGroup = new ec2.SecurityGroup(this, 'BatchComputeSg', {
      vpc: props.vpc,
      securityGroupName: `${props.projectName}-${props.environment}-batch-rl-sg`,
      description: 'Batch RL compute: self-referencing for NCCL + EFS',
      allowAllOutbound: true,
    });

    // Self-referencing rule: allow all traffic between compute nodes for NCCL
    // (NCCL uses dynamic ports and needs full mesh connectivity).
    this.securityGroup.addIngressRule(
      this.securityGroup,
      ec2.Port.allTraffic(),
      'NCCL inter-node communication (p2p GPU transfer)',
    );

    // NFS access to EFS (port 2049)
    this.securityGroup.addIngressRule(
      this.securityGroup,
      ec2.Port.tcp(2049),
      'EFS mount',
    );

    // IAM instance role for compute instances
    const instanceRole = new iam.Role(this, 'BatchInstanceRole', {
      roleName: `${props.projectName}-${props.environment}-batch-rl-instance-role`,
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [
        // ECS agent (Batch uses ECS under the hood)
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonEC2ContainerServiceforEC2Role'),
        // SSM Session Manager (for debugging)
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore'),
      ],
    });

    // S3 access for datasets/models (optional — add if your training reads from S3)
    // NOTE: The wildcard intentionally matches all project buckets (datasets, models, checkpoints).
    // For production, enumerate explicit bucket ARNs instead of the wildcard.
    instanceRole.addToPolicy(new iam.PolicyStatement({
      actions: ['s3:GetObject', 's3:PutObject', 's3:ListBucket'],
      resources: [
        `arn:aws:s3:::${props.projectName}-${props.environment}-*`,
        `arn:aws:s3:::${props.projectName}-${props.environment}-*/*`,
      ],
    }));

    // ECR pull access (Batch pulls the isaac-lab image)
    // GetAuthorizationToken is account-wide (AWS requires resources: ['*'])
    instanceRole.addToPolicy(new iam.PolicyStatement({
      actions: ['ecr:GetAuthorizationToken'],
      resources: ['*'],
    }));

    // Image pull actions scoped to the isaac-lab repo (least privilege)
    const region = cdk.Aws.REGION;
    const account = cdk.Aws.ACCOUNT_ID;
    instanceRole.addToPolicy(new iam.PolicyStatement({
      actions: [
        'ecr:BatchCheckLayerAvailability',
        'ecr:GetDownloadUrlForLayer',
        'ecr:BatchGetImage',
      ],
      resources: [`arn:aws:ecr:${region}:${account}:repository/${props.projectName}/isaac-lab`],
    }));

    const instanceProfile = new iam.CfnInstanceProfile(this, 'InstanceProfile', {
      instanceProfileName: `${props.projectName}-${props.environment}-batch-rl-profile`,
      roles: [instanceRole.roleName],
    });

    // Launch template: ECS-optimized GPU AMI + larger EBS
    // The AMI is resolved by Batch (it uses the latest ECS GPU-optimized AMI for the region).
    // We just set the block device mapping to provision a bigger root volume (250 GB gp3).
    const launchTemplate = new ec2.LaunchTemplate(this, 'LaunchTemplate', {
      launchTemplateName: `${props.projectName}-${props.environment}-batch-rl-lt`,
      instanceType: new ec2.InstanceType(instanceType),
      securityGroup: this.securityGroup,
      blockDevices: [{
        deviceName: '/dev/xvda',
        volume: ec2.BlockDeviceVolume.ebs(250, {
          volumeType: ec2.EbsDeviceVolumeType.GP3,
          encrypted: true,
        }),
      }],
    });

    // Compute environment: MANAGED EC2, GPU instances
    // NOTE: If using default VPC (no private subnets), place in public subnets.
    // For production, use a VPC with private subnets + NAT gateway.
    const hasPrivateSubnets = props.vpc.privateSubnets.length > 0;
    this.computeEnvironment = new batch.ManagedEc2EcsComputeEnvironment(this, 'ComputeEnv', {
      computeEnvironmentName: `${props.projectName}-${props.environment}-rl-compute`,
      vpc: props.vpc,
      vpcSubnets: hasPrivateSubnets
        ? { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS }
        : { subnetType: ec2.SubnetType.PUBLIC },
      instanceTypes: [new ec2.InstanceType(instanceType)],
      maxvCpus,
      // Use the launch template we created (Batch will pick the AMI)
      launchTemplate,
      // IMPORTANT: Batch needs an instance profile for the ECS agent + ECR/S3 access.
      // The CDK L2 doesn't expose instanceProfile, but the CFN escape hatch does:
      // computeEnvironmentName is set above, so we use addPropertyOverride.
      // Alternatively, pass via ComputeResources if the L2 adds it in the future.
    });

    // CFN escape hatch: attach the instance profile to the compute environment
    const cfnComputeEnv = this.computeEnvironment.node.defaultChild as batch.CfnComputeEnvironment;
    cfnComputeEnv.addPropertyOverride('ComputeResources.InstanceRole', instanceProfile.ref);

    // Job queue → compute environment
    this.queue = new batch.JobQueue(this, 'Queue', {
      jobQueueName: `${props.projectName}-${props.environment}-rl-queue`,
      computeEnvironments: [{
        computeEnvironment: this.computeEnvironment,
        order: 1,
      }],
    });

    // Multi-node job definition
    // NOTE: We request 4 GPUs + near-all vCPUs/memory (for g6.12xlarge: 48 vCPUs, 192 GB RAM)
    // so Batch places exactly ONE container per instance. This ensures full NCCL topology
    // visibility and avoids inter-container contention on the same node.
    const gpuCount = 4; // g6.12xlarge has 4x L4 GPUs
    const vCpus = 46;   // leave 2 vCPUs for the ECS agent
    const memoryMiB = 184 * 1024; // 184 GB (leave 8 GB for system)

    // Container definition
    const container = new batch.EcsEc2ContainerDefinition(this, 'Container', {
      image: ecs.ContainerImage.fromEcrRepository(props.isaacLabRepo, 'latest'),
      command: ['batch-train'],
      cpu: vCpus,
      memory: cdk.Size.mebibytes(memoryMiB),
      gpu: gpuCount,
      // Environment variables (hyperparameters; can be overridden per job via nodeOverrides)
      environment: {
        TASK: 'Isaac-Velocity-Flat-Anymal-D-v0',
        NUM_ENVS: '4096',
        MAX_ITERATIONS: '100',
        PROC_PER_NODE: gpuCount.toString(),
        FRAMEWORK: 'rsl_rl',
        ACCEPT_EULA: 'Y',
      },
      // EFS volume + mount
      volumes: [
        batch.EcsVolume.efs({
          name: 'efs',
          fileSystem: props.efsFileSystem,
          containerPath: '/efs',
        }),
      ],
      // Shared memory for Isaac Sim physics (64 GB)
      linuxParameters: new batch.LinuxParameters(this, 'LinuxParams', {
        sharedMemorySize: cdk.Size.gibibytes(64),
      }),
      // Logging to CloudWatch
      logging: ecs.LogDriver.awsLogs({ streamPrefix: 'batch-rl' }),
      // Job timeout (1 hour default; override per job if needed)
      jobRole: new iam.Role(this, 'JobRole', {
        roleName: `${props.projectName}-${props.environment}-batch-rl-job-role`,
        assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
        // Job role can be empty (the instance role handles S3/ECR); include if jobs
        // need to make AWS SDK calls directly (e.g., SageMaker, Bedrock).
      }),
    });

    // Multi-node job definition (2 nodes by default, single node group)
    this.jobDefinition = new batch.MultiNodeJobDefinition(this, 'JobDef', {
      jobDefinitionName: `${props.projectName}-${props.environment}-rl-mnp`,
      containers: [
        {
          container,
          startNode: 0,
          endNode: numNodes - 1,
        },
      ],
      instanceType: new ec2.InstanceType(instanceType),
      timeout: cdk.Duration.hours(1),
    });
  }
}
