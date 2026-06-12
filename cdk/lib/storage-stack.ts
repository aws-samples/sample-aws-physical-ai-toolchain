import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import { Construct } from 'constructs';

export interface StorageStackProps extends cdk.StackProps {
  environment: string;
  bucketPrefix: string;
}

/**
 * Storage stack: S3 buckets for pipeline data and ECR repos for containers.
 *
 * S3 Buckets:
 * - usd-assets: Cosmos-generated scenes (USD format)
 * - checkpoints: Training checkpoints (saved every N steps)
 * - models: Final exported models (ONNX + TensorRT engines)
 * - telemetry: Edge device telemetry (failed grasps, metrics)
 *
 * ECR Repositories:
 * - isaac-sim: Extended Isaac Sim container with custom configs
 * - isaac-lab: Training container with RL dependencies
 * - inference: TensorRT + ROS 2 inference node for edge
 */
export class StorageStack extends cdk.Stack {
  public readonly assetsBucket: s3.Bucket;
  public readonly checkpointsBucket: s3.Bucket;
  public readonly modelsBucket: s3.Bucket;
  public readonly telemetryBucket: s3.Bucket;
  public readonly isaacSimRepo: ecr.Repository;
  public readonly isaacLabRepo: ecr.Repository;
  public readonly inferenceRepo: ecr.Repository;

  constructor(scope: Construct, id: string, props: StorageStackProps) {
    super(scope, id, props);

    const { bucketPrefix } = props;

    // --- S3 Buckets ---

    // USD scene assets (Cosmos-generated + hand-crafted)
    this.assetsBucket = new s3.Bucket(this, 'AssetsBucket', {
      bucketName: `${bucketPrefix}-usd-assets`,
      versioned: false, // Scenes are regenerated, not versioned
      lifecycleRules: [
        {
          // Clean up old generated scenes after 30 days
          expiration: cdk.Duration.days(30),
          prefix: 'generated/',
        },
      ],
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    // Training checkpoints (intermediate model saves)
    this.checkpointsBucket = new s3.Bucket(this, 'CheckpointsBucket', {
      bucketName: `${bucketPrefix}-checkpoints`,
      versioned: false,
      lifecycleRules: [
        {
          // Keep only last 7 days of checkpoints (they're large)
          expiration: cdk.Duration.days(7),
        },
      ],
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    // Final exported models (ONNX + TensorRT)
    this.modelsBucket = new s3.Bucket(this, 'ModelsBucket', {
      bucketName: `${bucketPrefix}-models`,
      versioned: true, // Version models for rollback
      removalPolicy: cdk.RemovalPolicy.RETAIN, // Don't delete trained models
    });

    // Edge telemetry (failed grasps, performance metrics)
    this.telemetryBucket = new s3.Bucket(this, 'TelemetryBucket', {
      bucketName: `${bucketPrefix}-telemetry`,
      versioned: false,
      lifecycleRules: [
        {
          // Move to Glacier after 90 days
          transitions: [
            {
              storageClass: s3.StorageClass.GLACIER,
              transitionAfter: cdk.Duration.days(90),
            },
          ],
        },
      ],
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    // --- ECR Repositories ---

    this.isaacSimRepo = new ecr.Repository(this, 'IsaacSimRepo', {
      repositoryName: `${props.environment}/isaac-sim`,
      imageScanOnPush: true,
      lifecycleRules: [
        { maxImageCount: 5, description: 'Keep last 5 images' },
      ],
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      emptyOnDelete: true,
    });

    this.isaacLabRepo = new ecr.Repository(this, 'IsaacLabRepo', {
      repositoryName: `${props.environment}/isaac-lab`,
      imageScanOnPush: true,
      lifecycleRules: [
        { maxImageCount: 5, description: 'Keep last 5 images' },
      ],
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      emptyOnDelete: true,
    });

    this.inferenceRepo = new ecr.Repository(this, 'InferenceRepo', {
      repositoryName: `${props.environment}/inference`,
      imageScanOnPush: true,
      lifecycleRules: [
        { maxImageCount: 10, description: 'Keep last 10 images' },
      ],
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      emptyOnDelete: true,
    });

    // --- Outputs ---
    new cdk.CfnOutput(this, 'AssetsBucketName', { value: this.assetsBucket.bucketName });
    new cdk.CfnOutput(this, 'ModelsBucketName', { value: this.modelsBucket.bucketName });
    new cdk.CfnOutput(this, 'IsaacSimRepoUri', { value: this.isaacSimRepo.repositoryUri });
    new cdk.CfnOutput(this, 'IsaacLabRepoUri', { value: this.isaacLabRepo.repositoryUri });
    new cdk.CfnOutput(this, 'InferenceRepoUri', { value: this.inferenceRepo.repositoryUri });
  }
}
