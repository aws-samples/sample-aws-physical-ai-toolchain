import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { Construct } from 'constructs';

export interface NetworkStackProps extends cdk.StackProps {
  environment: string;
}

/**
 * Network stack: VPC with public/private subnets for EKS, RDS, and ElastiCache.
 *
 * Architecture:
 * - Public subnets: NAT Gateways, load balancers (OSMO UI)
 * - Private subnets: EKS nodes (GPU + control), RDS, ElastiCache
 * - Isolated subnets: Not used (everything needs outbound for NGC container pulls)
 */
export class NetworkStack extends cdk.Stack {
  public readonly vpc: ec2.Vpc;
  public readonly eksSecurityGroup: ec2.SecurityGroup;
  public readonly dbSecurityGroup: ec2.SecurityGroup;
  public readonly redisSecurityGroup: ec2.SecurityGroup;

  constructor(scope: Construct, id: string, props: NetworkStackProps) {
    super(scope, id, props);

    // VPC with 2 AZs (sufficient for dev, prod can override)
    this.vpc = new ec2.Vpc(this, 'PhysicalAiVpc', {
      vpcName: `physical-ai-${props.environment}-vpc`,
      maxAzs: 2,
      natGateways: 1, // Single NAT for dev cost savings; prod should use 2
      subnetConfiguration: [
        {
          cidrMask: 24,
          name: 'Public',
          subnetType: ec2.SubnetType.PUBLIC,
        },
        {
          cidrMask: 20, // Large range for EKS pods (GPU nodes need many IPs)
          name: 'Private',
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
        },
      ],
    });

    // Security group for EKS cluster and nodes
    this.eksSecurityGroup = new ec2.SecurityGroup(this, 'EksSecurityGroup', {
      vpc: this.vpc,
      securityGroupName: `physical-ai-${props.environment}-eks-sg`,
      description: 'Security group for EKS cluster nodes',
      allowAllOutbound: true, // Nodes need outbound for NGC pulls, S3, ECR
    });

    // Security group for RDS (OSMO PostgreSQL)
    this.dbSecurityGroup = new ec2.SecurityGroup(this, 'DbSecurityGroup', {
      vpc: this.vpc,
      securityGroupName: `physical-ai-${props.environment}-db-sg`,
      description: 'Security group for OSMO PostgreSQL database',
      allowAllOutbound: false,
    });

    // Allow EKS → RDS on port 5432
    // Note: EKS creates its own cluster security group in addition to the one we provide.
    // Pods use the EKS-managed SG, so we allow from both.
    this.dbSecurityGroup.addIngressRule(
      this.eksSecurityGroup,
      ec2.Port.tcp(5432),
      'Allow EKS custom SG to connect to PostgreSQL'
    );
    // The EKS-managed cluster SG will be added post-deploy via:
    // aws ec2 authorize-security-group-ingress --group-id <db-sg> --protocol tcp --port 5432 --source-group <eks-cluster-sg>
    // TODO: Automate this with a custom resource or by looking up the cluster SG after creation

    // Security group for ElastiCache (OSMO Redis)
    this.redisSecurityGroup = new ec2.SecurityGroup(this, 'RedisSecurityGroup', {
      vpc: this.vpc,
      securityGroupName: `physical-ai-${props.environment}-redis-sg`,
      description: 'Security group for OSMO Redis cache',
      allowAllOutbound: false,
    });

    // Allow EKS → Redis on port 6379
    this.redisSecurityGroup.addIngressRule(
      this.eksSecurityGroup,
      ec2.Port.tcp(6379),
      'Allow EKS custom SG to connect to Redis'
    );
    // Same note as above — EKS-managed cluster SG also needs access

    // VPC Endpoints for cost savings and performance
    // S3 Gateway endpoint (free, avoids NAT charges for S3 traffic)
    this.vpc.addGatewayEndpoint('S3Endpoint', {
      service: ec2.GatewayVpcEndpointAwsService.S3,
    });

    // ECR endpoints (avoids NAT for container pulls)
    this.vpc.addInterfaceEndpoint('EcrEndpoint', {
      service: ec2.InterfaceVpcEndpointAwsService.ECR,
    });
    this.vpc.addInterfaceEndpoint('EcrDockerEndpoint', {
      service: ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER,
    });

    // Outputs
    new cdk.CfnOutput(this, 'VpcId', { value: this.vpc.vpcId });
  }
}
