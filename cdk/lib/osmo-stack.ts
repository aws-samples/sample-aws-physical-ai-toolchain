import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as rds from 'aws-cdk-lib/aws-rds';
import * as elasticache from 'aws-cdk-lib/aws-elasticache';
import * as eks from 'aws-cdk-lib/aws-eks';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import { Construct } from 'constructs';

export interface OsmoStackProps extends cdk.StackProps {
  environment: string;
  vpc: ec2.Vpc;
  dbSecurityGroup: ec2.SecurityGroup;
  redisSecurityGroup: ec2.SecurityGroup;
  cluster: eks.Cluster;
  osmoConfig: {
    postgres: {
      instanceType: string;
      allocatedStorage: number;
      multiAz: boolean;
    };
    redis: {
      nodeType: string;
      numNodes: number;
    };
  };
}

/**
 * OSMO Stack: Deploys OSMO control plane dependencies and the OSMO Helm chart.
 *
 * Components:
 * - RDS PostgreSQL: Stores workflow metadata, job state, scheduling info
 * - ElastiCache Redis: Caching, session management, pub/sub for operators
 * - OSMO Helm chart: API server, scheduler, web UI deployed to EKS
 *
 * OSMO architecture:
 * - Control plane runs on the 'control' node pool (non-GPU)
 * - Operators (compute agents) run on GPU nodes and edge devices
 * - Edge operators connect outbound to control plane (no inbound rules needed)
 */
export class OsmoStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: OsmoStackProps) {
    super(scope, id, props);

    const { vpc, dbSecurityGroup, redisSecurityGroup, cluster, osmoConfig } = props;

    // --- PostgreSQL for OSMO ---
    const dbSecret = new secretsmanager.Secret(this, 'OsmoDbSecret', {
      secretName: `physical-ai-${props.environment}/osmo-db`,
      generateSecretString: {
        secretStringTemplate: JSON.stringify({ username: 'osmo' }),
        generateStringKey: 'password',
        excludePunctuation: true,
        passwordLength: 32,
      },
    });

    const dbInstance = new rds.DatabaseInstance(this, 'OsmoPostgres', {
      engine: rds.DatabaseInstanceEngine.postgres({
        version: rds.PostgresEngineVersion.VER_15_10,
      }),
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.T4G, ec2.InstanceSize.MEDIUM),
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      securityGroups: [dbSecurityGroup],
      databaseName: 'osmo',
      credentials: rds.Credentials.fromSecret(dbSecret),
      allocatedStorage: osmoConfig.postgres.allocatedStorage,
      multiAz: osmoConfig.postgres.multiAz,
      storageEncrypted: true,
      backupRetention: cdk.Duration.days(7),
      deletionProtection: props.environment === 'prod',
      removalPolicy: props.environment === 'prod'
        ? cdk.RemovalPolicy.RETAIN
        : cdk.RemovalPolicy.DESTROY,
    });

    // --- Redis for OSMO ---
    const redisSubnetGroup = new elasticache.CfnSubnetGroup(this, 'RedisSubnetGroup', {
      description: 'Subnet group for OSMO Redis',
      subnetIds: vpc.selectSubnets({ subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS }).subnetIds,
      cacheSubnetGroupName: `physical-ai-${props.environment}-redis`,
    });

    const redisCluster = new elasticache.CfnCacheCluster(this, 'OsmoRedis', {
      engine: 'redis',
      cacheNodeType: osmoConfig.redis.nodeType,
      numCacheNodes: osmoConfig.redis.numNodes,
      vpcSecurityGroupIds: [redisSecurityGroup.securityGroupId],
      cacheSubnetGroupName: redisSubnetGroup.cacheSubnetGroupName,
      engineVersion: '7.1',
      clusterName: `osmo-${props.environment}`,
    });
    redisCluster.addDependency(redisSubnetGroup);

    // --- OSMO Namespace ---
    // Will be created manually after cluster is up:
    //   kubectl create namespace osmo
    //   kubectl apply -f https://raw.githubusercontent.com/NVIDIA/OSMO/main/deployments/k8s/

    // --- OSMO Helm Chart ---
    // OSMO's Helm repo is not yet publicly available.
    // Install manually after cluster is up using raw manifests from GitHub:
    //   git clone https://github.com/NVIDIA/OSMO.git
    //   kubectl apply -f OSMO/deployments/k8s/ -n osmo
    //
    // Or when Helm repo becomes available:
    //   helm repo add osmo https://nvidia.github.io/OSMO/helm-charts
    //   helm install osmo osmo/osmo -n osmo --set database.host=<RDS_ENDPOINT> --set redis.host=<REDIS_ENDPOINT>

    // --- Outputs ---
    new cdk.CfnOutput(this, 'PostgresEndpoint', {
      value: dbInstance.instanceEndpoint.hostname,
    });
    new cdk.CfnOutput(this, 'RedisEndpoint', {
      value: redisCluster.attrRedisEndpointAddress,
    });
    new cdk.CfnOutput(this, 'OsmoUiAccess', {
      value: 'kubectl port-forward -n osmo svc/osmo-ui 8080:80',
      description: 'Run this to access OSMO Web UI at http://localhost:8080',
    });
  }
}
