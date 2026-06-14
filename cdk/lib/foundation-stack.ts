import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as codebuild from 'aws-cdk-lib/aws-codebuild';
import { Construct } from 'constructs';

export interface FoundationStackProps extends cdk.StackProps {
  environment: string;
  projectName: string;
}

/**
 * Foundation Stack — always deployed regardless of mode (simple or full).
 *
 * Provides the shared resources that both Path A (SageMaker) and Path B (EKS/OSMO) need:
 * - S3 buckets for datasets, models, checkpoints
 * - ECR repositories for training and inference containers
 * - IAM roles for SageMaker and CodeBuild
 * - CodeBuild projects for container image builds
 *
 * WORKSHOP NOTE: This is the first thing you deploy. It takes ~3 minutes.
 * After deployment, check the CloudFormation outputs for bucket names and role ARNs.
 */
export class FoundationStack extends cdk.Stack {
  // Expose resources for other stacks to reference
  public readonly datasetsBucket: s3.Bucket;
  public readonly modelsBucket: s3.Bucket;
  public readonly checkpointsBucket: s3.Bucket;
  public readonly grootTrainingRepo: ecr.Repository;
  public readonly isaacLabRepo: ecr.Repository;
  public readonly inferenceRepo: ecr.Repository;
  public readonly sagemakerRole: iam.Role;

  constructor(scope: Construct, id: string, props: FoundationStackProps) {
    super(scope, id, props);

    const { projectName, environment } = props;
    const account = cdk.Aws.ACCOUNT_ID;
    const region = cdk.Aws.REGION;

    // =========================================================================
    // S3 BUCKETS
    // =========================================================================

    // Training datasets (LeRobot format for GR00T, USD for Isaac Lab)
    this.datasetsBucket = new s3.Bucket(this, 'DatasetsBucket', {
      bucketName: `${projectName}-${environment}-datasets-${account}`,
      versioned: true,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      removalPolicy: environment === 'prod'
        ? cdk.RemovalPolicy.RETAIN
        : cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: environment !== 'prod',
    });

    // Trained models (ONNX, TensorRT, model.tar.gz)
    this.modelsBucket = new s3.Bucket(this, 'ModelsBucket', {
      bucketName: `${projectName}-${environment}-models-${account}`,
      versioned: true, // Version for rollback
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      removalPolicy: cdk.RemovalPolicy.RETAIN, // Never auto-delete trained models
    });

    // Training checkpoints (intermediate, can be cleaned up)
    this.checkpointsBucket = new s3.Bucket(this, 'CheckpointsBucket', {
      bucketName: `${projectName}-${environment}-checkpoints-${account}`,
      versioned: false,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      lifecycleRules: [
        { expiration: cdk.Duration.days(14) }, // Auto-clean after 2 weeks
      ],
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    // =========================================================================
    // ECR REPOSITORIES
    // =========================================================================

    // GR00T fine-tuning container (Path A)
    this.grootTrainingRepo = new ecr.Repository(this, 'GrootTrainingRepo', {
      repositoryName: `${projectName}/groot-training`,
      imageScanOnPush: true,
      lifecycleRules: [{ maxImageCount: 5 }],
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      emptyOnDelete: true,
    });

    // Isaac Lab RL training container (Path B)
    this.isaacLabRepo = new ecr.Repository(this, 'IsaacLabRepo', {
      repositoryName: `${projectName}/isaac-lab`,
      imageScanOnPush: true,
      lifecycleRules: [{ maxImageCount: 5 }],
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      emptyOnDelete: true,
    });

    // Inference container — TensorRT + ROS2 (shared, used by edge)
    this.inferenceRepo = new ecr.Repository(this, 'InferenceRepo', {
      repositoryName: `${projectName}/inference`,
      imageScanOnPush: true,
      lifecycleRules: [{ maxImageCount: 10 }],
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      emptyOnDelete: true,
    });

    // =========================================================================
    // IAM: SAGEMAKER EXECUTION ROLE
    // =========================================================================

    this.sagemakerRole = new iam.Role(this, 'SageMakerExecutionRole', {
      roleName: `${projectName}-${environment}-sagemaker-role`,
      assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSageMakerFullAccess'),
      ],
    });

    // S3 access for all buckets
    this.datasetsBucket.grantReadWrite(this.sagemakerRole);
    this.modelsBucket.grantReadWrite(this.sagemakerRole);
    this.checkpointsBucket.grantReadWrite(this.sagemakerRole);

    // ECR pull access for training containers
    this.sagemakerRole.addToPolicy(new iam.PolicyStatement({
      actions: [
        'ecr:GetAuthorizationToken',
        'ecr:BatchCheckLayerAvailability',
        'ecr:GetDownloadUrlForLayer',
        'ecr:BatchGetImage',
      ],
      resources: ['*'],
    }));

    // =========================================================================
    // CODEBUILD: CONTAINER BUILD PROJECTS
    // =========================================================================

    const codebuildRole = new iam.Role(this, 'CodeBuildRole', {
      roleName: `${projectName}-${environment}-codebuild-role`,
      assumedBy: new iam.ServicePrincipal('codebuild.amazonaws.com'),
    });

    // ECR push permissions
    codebuildRole.addToPolicy(new iam.PolicyStatement({
      actions: [
        'ecr:GetAuthorizationToken',
        'ecr:BatchCheckLayerAvailability',
        'ecr:GetDownloadUrlForLayer',
        'ecr:BatchGetImage',
        'ecr:InitiateLayerUpload',
        'ecr:UploadLayerPart',
        'ecr:CompleteLayerUpload',
        'ecr:PutImage',
      ],
      resources: ['*'],
    }));

    codebuildRole.addToPolicy(new iam.PolicyStatement({
      actions: ['logs:CreateLogGroup', 'logs:CreateLogStream', 'logs:PutLogEvents'],
      resources: ['*'],
    }));

    this.datasetsBucket.grantRead(codebuildRole);

    // GR00T training container build project
    new codebuild.Project(this, 'GrootTrainingBuild', {
      projectName: `${projectName}-groot-training-build`,
      role: codebuildRole,
      environment: {
        buildImage: codebuild.LinuxBuildImage.STANDARD_7_0,
        computeType: codebuild.ComputeType.LARGE,
        privileged: true,
        environmentVariables: {
          AWS_DEFAULT_REGION: { value: region },
          AWS_ACCOUNT_ID: { value: account },
          ECR_REPO: { value: this.grootTrainingRepo.repositoryUri },
        },
      },
      source: codebuild.Source.gitHub({
        owner: 'PLACEHOLDER', // Will be updated when repo is public
        repo: 'aws-physical-ai-toolchain',
        branchOrRef: 'main',
      }),
      buildSpec: codebuild.BuildSpec.fromObject({
        version: '0.2',
        phases: {
          pre_build: {
            commands: [
              'aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com',
            ],
          },
          build: {
            commands: [
              'cd containers/groot-training',
              'docker build -t groot-training .',
              'docker tag groot-training:latest $ECR_REPO:latest',
            ],
          },
          post_build: {
            commands: ['docker push $ECR_REPO:latest'],
          },
        },
      }),
    });

    // Isaac Lab RL training container build project
    // Requires NGC API key stored in Secrets Manager (for base image pull)
    // Uses x86 large instance (image is ~30GB, needs space + time)
    codebuildRole.addToPolicy(new iam.PolicyStatement({
      actions: ['secretsmanager:GetSecretValue'],
      resources: [`arn:aws:secretsmanager:${region}:${account}:secret:${projectName}/ngc-api-key*`],
    }));

    new codebuild.Project(this, 'IsaacLabBuild', {
      projectName: `${projectName}-isaac-lab-build`,
      role: codebuildRole,
      environment: {
        buildImage: codebuild.LinuxBuildImage.STANDARD_7_0,
        computeType: codebuild.ComputeType.X2_LARGE, // 72 GB memory, 300 GB disk for large image
        privileged: true,
        environmentVariables: {
          AWS_DEFAULT_REGION: { value: region },
          ECR_REPO_URI: { value: this.isaacLabRepo.repositoryUri },
        },
      },
      source: codebuild.Source.gitHub({
        owner: 'PLACEHOLDER',
        repo: 'aws-physical-ai-toolchain',
        branchOrRef: 'main',
      }),
      buildSpec: codebuild.BuildSpec.fromSourceFilename('containers/isaac-lab/buildspec.yml'),
      timeout: cdk.Duration.hours(2), // Large image build can take a while
    });

    // =========================================================================
    // OUTPUTS
    // =========================================================================

    new cdk.CfnOutput(this, 'DatasetsBucketName', {
      value: this.datasetsBucket.bucketName,
      description: 'Upload training datasets here (LeRobot format)',
      exportName: `${projectName}-${environment}-datasets-bucket`,
    });

    new cdk.CfnOutput(this, 'ModelsBucketName', {
      value: this.modelsBucket.bucketName,
      description: 'Trained models are written here',
      exportName: `${projectName}-${environment}-models-bucket`,
    });

    new cdk.CfnOutput(this, 'SageMakerRoleArn', {
      value: this.sagemakerRole.roleArn,
      description: 'Pass this as the execution role for SageMaker training jobs',
      exportName: `${projectName}-${environment}-sagemaker-role-arn`,
    });

    new cdk.CfnOutput(this, 'GrootTrainingRepoUri', {
      value: this.grootTrainingRepo.repositoryUri,
      description: 'ECR URI for the GR00T training container',
      exportName: `${projectName}-${environment}-groot-training-ecr`,
    });

    new cdk.CfnOutput(this, 'InferenceRepoUri', {
      value: this.inferenceRepo.repositoryUri,
      description: 'ECR URI for the inference container (edge deployment)',
      exportName: `${projectName}-${environment}-inference-ecr`,
    });

    new cdk.CfnOutput(this, 'IsaacLabRepoUri', {
      value: this.isaacLabRepo.repositoryUri,
      description: 'ECR URI for the Isaac Lab RL training container',
      exportName: `${projectName}-${environment}-isaac-lab-ecr`,
    });
  }
}
