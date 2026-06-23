#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { FoundationStack } from '../lib/foundation-stack';
import { NetworkStack } from '../lib/network-stack';
import { StorageStack } from '../lib/storage-stack';
import { EksClusterStack } from '../lib/eks-cluster-stack';
import { OsmoStack } from '../lib/osmo-stack';
import { EdgeStack } from '../lib/edge-stack';
import { WorkstationStack } from '../lib/workstation-stack';
import { devConfig } from '../config/dev';
import { prodConfig } from '../config/prod';
import * as fs from 'fs';
import * as path from 'path';

/**
 * AWS Physical AI Toolchain — CDK Application
 *
 * Two deployment modes:
 *
 *   SIMPLE (Path A):  Foundation only → GR00T training on SageMaker
 *     cdk deploy --context mode=simple
 *
 *   FULL (Path B):    Foundation + EKS + OSMO → Isaac Lab RL training
 *     cdk deploy --context mode=full
 *
 * Both modes deploy the Foundation stack (S3, ECR, IAM, CodeBuild).
 * Full mode adds EKS, OSMO, and the network infrastructure.
 * The Edge stack (Greengrass) is optional and works with either mode.
 *
 * WORKSHOP NOTE:
 *   Lab 1 uses mode=simple. Lab 2 upgrades to mode=full.
 *   You can always add Path B later without rebuilding Path A.
 */
const app = new cdk.App();

// Load root-level config.json (workstation settings, AMI mapping, etc.)
const rootConfig = JSON.parse(
  fs.readFileSync(path.join(__dirname, '../../config.json'), 'utf-8')
);

// --- Configuration ---
const envName = app.node.tryGetContext('env') || 'dev';
const mode = app.node.tryGetContext('mode') || 'simple'; // 'simple' | 'full'
const includeEdge = app.node.tryGetContext('edge') === 'true'; // opt-in
const includeWorkstation = app.node.tryGetContext('workstation') === 'true'; // opt-in

const config = envName === 'prod' ? prodConfig : devConfig;
const projectName = 'physical-ai';

// Region resolution — config.json is the single source of truth (fixes F-006 region drift,
// where the shell's AWS_REGION/CDK_DEFAULT_REGION could silently win over the documented region).
// Precedence: explicit --context region= override > config.json aws.region > shell default > us-east-1.
const region =
  app.node.tryGetContext('region') ||
  rootConfig.aws?.region ||
  process.env.CDK_DEFAULT_REGION ||
  'us-east-1';

const env: cdk.Environment = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region,
};

const prefix = `PhysicalAi-${config.environment}`;

console.log(`\n  Physical AI Toolchain`);
console.log(`  Mode: ${mode}  |  Env: ${envName}  |  Edge: ${includeEdge}`);
console.log(`  Region: ${env.region}\n`);

// ═══════════════════════════════════════════════════════════════════════════════
// FOUNDATION (always deployed — both modes)
// ═══════════════════════════════════════════════════════════════════════════════

const foundationStack = new FoundationStack(app, `${prefix}-Foundation`, {
  env,
  environment: config.environment,
  projectName,
});

// ═══════════════════════════════════════════════════════════════════════════════
// PATH B: FULL MODE (EKS + OSMO) — only when mode=full
// ═══════════════════════════════════════════════════════════════════════════════

if (mode === 'full') {
  // Network (VPC, subnets, security groups, VPC endpoints)
  const networkStack = new NetworkStack(app, `${prefix}-Network`, {
    env,
    environment: config.environment,
  });

  // Storage (legacy S3 + ECR for OSMO — supplements foundation)
  const storageStack = new StorageStack(app, `${prefix}-Storage`, {
    env,
    environment: config.environment,
    bucketPrefix: config.storage.bucketPrefix,
  });

  // EKS Cluster (GPU + control node pools)
  const eksStack = new EksClusterStack(app, `${prefix}-Eks`, {
    env,
    environment: config.environment,
    vpc: networkStack.vpc,
    eksSecurityGroup: networkStack.eksSecurityGroup,
    eksConfig: config.eks,
    assetsBucket: storageStack.assetsBucket,
    checkpointsBucket: storageStack.checkpointsBucket,
    modelsBucket: storageStack.modelsBucket,
    telemetryBucket: storageStack.telemetryBucket,
    // Pull from the Foundation ECR repos — those are the ones the CodeBuild
    // jobs actually populate. (StorageStack used to declare its own empty
    // isaac-*/inference repos that nothing ever pushed to.)
    isaacSimRepo: foundationStack.isaacSimRepo,
    isaacLabRepo: foundationStack.isaacLabRepo,
    inferenceRepo: foundationStack.inferenceRepo,
  });

  // OSMO control plane (RDS, Redis, Helm chart on EKS)
  const osmoStack = new OsmoStack(app, `${prefix}-Osmo`, {
    env,
    environment: config.environment,
    vpc: networkStack.vpc,
    dbSecurityGroup: networkStack.dbSecurityGroup,
    redisSecurityGroup: networkStack.redisSecurityGroup,
    cluster: eksStack.cluster,
    osmoConfig: config.osmo,
  });

  // Dependencies
  eksStack.addDependency(networkStack);
  eksStack.addDependency(storageStack);
  eksStack.addDependency(foundationStack); // EKS pulls images from Foundation ECR repos
  osmoStack.addDependency(eksStack);
}

// ═══════════════════════════════════════════════════════════════════════════════
// EDGE (optional — works with either mode)
// ═══════════════════════════════════════════════════════════════════════════════

if (includeEdge) {
  const edgeStack = new EdgeStack(app, `${prefix}-Edge`, {
    env,
    environment: config.environment,
    thingGroupName: config.edge.thingGroupName,
    modelsBucket: foundationStack.modelsBucket,
    telemetryBucket: foundationStack.datasetsBucket, // Reuse datasets bucket for telemetry in simple mode
  });

  edgeStack.addDependency(foundationStack);
}

// ═══════════════════════════════════════════════════════════════════════════════
// WORKSTATION (optional — Isaac Sim dev environment with DCV)
// ═══════════════════════════════════════════════════════════════════════════════

if (includeWorkstation) {
  // Context flags override config.json for backward-compatibility
  const allowedCidr = app.node.tryGetContext('allowedCidr') || rootConfig.workstation.allowedCidr; // no default — WorkstationStack throws if missing (avoids silent 0.0.0.0/0)
  const instanceType = app.node.tryGetContext('instanceType') || rootConfig.workstation.instanceType;
  const repoUrl = app.node.tryGetContext('repoUrl'); // optional override
  const availabilityZone = app.node.tryGetContext('availabilityZone'); // optional AZ pin

  new WorkstationStack(app, `${prefix}-Workstation`, {
    env,
    environment: config.environment,
    allowedCidr,
    instanceType,
    amiMapping: rootConfig.workstation.amiMapping,
    volumeSizeGb: rootConfig.workstation.volumeSizeGb,
    dcvPort: rootConfig.workstation.dcvPort,
    projectName,
    repoUrl,
    availabilityZone,
  });
}

app.synth();
