import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as efs from 'aws-cdk-lib/aws-efs';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import { Construct } from 'constructs';
import { BatchRl } from './constructs/batch-rl';

export interface BatchStackProps extends cdk.StackProps {
  environment: string;
  projectName: string;
  /**
   * EC2 instance type for Batch compute nodes.
   * Default: g6.12xlarge (4x L4 GPUs).
   */
  instanceType?: string;
  /**
   * Number of nodes for multi-node parallel jobs.
   * Default: 2.
   */
  numNodes?: number;
  /**
   * Max vCPUs for the Batch compute environment (controls fleet size).
   * Default: 96 (2x g6.12xlarge).
   */
  maxvCpus?: number;
}

/**
 * Batch Stack — AWS Batch Multi-Node Parallel (MNP) RL training.
 *
 * Opt-in stack (deploy with --context batch=true). Provisions:
 * - EFS filesystem (shared checkpoints across nodes + persistence)
 * - Batch compute environment + job queue + MNP job definition
 *
 * Uses the default VPC (like WorkstationStack) for opt-in independence. If your
 * account has no default VPC, deploy fails — create one or adapt to use the
 * NetworkStack VPC (requires mode=full).
 *
 * Instance type, numNodes, and maxvCpus are read from config.json with
 * --context overrides supported.
 *
 * Deploy:
 *   cdk deploy PhysicalAi-dev-Foundation --context mode=simple  # builds isaac-lab image
 *   # Wait for CodeBuild to complete (~1 hour for isaac-lab)
 *   cdk deploy PhysicalAi-dev-Batch --context batch=true
 *
 * Submit a job:
 *   python training/scripts/launch_rl_batch.py \
 *     --task Isaac-Velocity-Flat-Anymal-D-v0 \
 *     --num-envs 4096 --max-iterations 100 --num-nodes 2
 *
 * NOTE: Multi-node NCCL convergence is UNVALIDATED on hardware. This stack
 * wires the topology correctly (inter-node SG rules, EFS, Batch MNP env vars),
 * but end-to-end distributed training has not been verified on g6 instances.
 * See plans/distributed-rl-and-eval/02-batch-mnp.md for known risks.
 */
export class BatchStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: BatchStackProps) {
    super(scope, id, props);

    const { projectName, environment } = props;
    const region = cdk.Aws.REGION;
    const account = cdk.Aws.ACCOUNT_ID;

    // Context overrides for instance type, numNodes, maxvCpus (from config.json or --context)
    const instanceType =
      this.node.tryGetContext('instanceType') || props.instanceType || 'g6.12xlarge';
    const numNodes =
      parseInt(this.node.tryGetContext('numNodes') || props.numNodes?.toString() || '2', 10);
    const maxvCpus =
      parseInt(this.node.tryGetContext('maxvCpus') || props.maxvCpus?.toString() || '96', 10);

    // Use default VPC for opt-in independence (same tradeoff as WorkstationStack).
    // RISK: If the account has no default VPC, this fails. Users can create one
    // or adapt to use the NetworkStack VPC (requires mode=full + cross-stack ref).
    const vpc = ec2.Vpc.fromLookup(this, 'DefaultVpc', { isDefault: true });

    // EFS filesystem for shared checkpoints (encrypted, in the same VPC)
    const efsSecurityGroup = new ec2.SecurityGroup(this, 'EfsSg', {
      vpc,
      securityGroupName: `${projectName}-${environment}-efs-sg`,
      description: 'EFS security group (allows NFS from Batch compute)',
      allowAllOutbound: false,
    });

    const fileSystem = new efs.FileSystem(this, 'EfsFileSystem', {
      vpc,
      fileSystemName: `${projectName}-${environment}-batch-rl-efs`,
      encrypted: true,
      securityGroup: efsSecurityGroup,
      removalPolicy: cdk.RemovalPolicy.DESTROY, // dev-friendly; change to RETAIN for prod
    });

    // Pull the isaac-lab ECR repo (built by FoundationStack)
    const isaacLabRepo = ecr.Repository.fromRepositoryName(
      this,
      'IsaacLabRepo',
      `${projectName}/isaac-lab`,
    );

    // Batch RL construct (compute env + queue + job def + EFS mount)
    const batchRl = new BatchRl(this, 'BatchRl', {
      vpc,
      projectName,
      environment,
      instanceType,
      isaacLabRepo,
      efsFileSystem: fileSystem,
      maxvCpus,
      numNodes,
    });

    // Allow Batch compute nodes to mount EFS (NFS port 2049)
    efsSecurityGroup.addIngressRule(
      batchRl.securityGroup,
      ec2.Port.tcp(2049),
      'Allow Batch compute to mount EFS',
    );

    // Outputs
    new cdk.CfnOutput(this, 'JobQueueName', {
      value: batchRl.queue.jobQueueName,
      description: 'AWS Batch job queue (pass to launch_rl_batch.py)',
      exportName: `${projectName}-${environment}-batch-rl-queue`,
    });

    new cdk.CfnOutput(this, 'JobDefinitionName', {
      value: batchRl.jobDefinition.jobDefinitionName,
      description: 'AWS Batch multi-node job definition',
      exportName: `${projectName}-${environment}-batch-rl-job-def`,
    });

    new cdk.CfnOutput(this, 'EfsFileSystemId', {
      value: fileSystem.fileSystemId,
      description: 'EFS filesystem ID (checkpoints persist here)',
      exportName: `${projectName}-${environment}-batch-rl-efs`,
    });

    new cdk.CfnOutput(this, 'LaunchCommand', {
      value: `python training/scripts/launch_rl_batch.py --task Isaac-Velocity-Flat-Anymal-D-v0 --num-envs 4096 --max-iterations 100 --num-nodes ${numNodes} --job-queue ${batchRl.queue.jobQueueName} --job-definition ${batchRl.jobDefinition.jobDefinitionName}`,
      description: 'Example launch_rl_batch.py command (paste this into your shell)',
    });

    new cdk.CfnOutput(this, 'MonitorCommand', {
      value: `aws batch describe-jobs --jobs <job-id> --region ${region}`,
      description: 'Monitor job status (replace <job-id> with the submitted job ID)',
    });

    new cdk.CfnOutput(this, 'Cost', {
      value: `~$6.00/hr (${instanceType} x ${numNodes}, on-demand us-west-2). Jobs auto-terminate on completion. EFS: ~$0.30/GB-month (pay only for stored data). Actual cost varies by region.`,
      description: 'Estimated cost — jobs only run when submitted',
    });
  }
}
