import * as cdk from 'aws-cdk-lib';
import * as iot from 'aws-cdk-lib/aws-iot';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as greengrassv2 from 'aws-cdk-lib/aws-greengrassv2';
import { Construct } from 'constructs';

export interface EdgeStackProps extends cdk.StackProps {
  environment: string;
  thingGroupName: string;
  modelsBucket: s3.Bucket;
  telemetryBucket: s3.Bucket;
  /**
   * ECR repo holding the edge inference image. Its URI is baked into the
   * Greengrass component recipe so devices pull the right container.
   */
  inferenceRepo: ecr.Repository;
}

/**
 * Edge Stack: IoT Core + Greengrass configuration for Jetson/GPU PC fleet management.
 *
 * Responsibilities:
 * - Device provisioning (IoT Thing + certificates)
 * - Greengrass component deployment (inference node, telemetry collector)
 * - OTA model updates (pull new TensorRT engines from S3)
 * - Health monitoring and device shadows
 *
 * Note: OSMO handles workflow execution on edge (deploy model, run inference).
 * Greengrass handles fleet operations (OTA, health, device management at scale).
 * They complement each other — OSMO for "what to run", Greengrass for "how to manage."
 */
export class EdgeStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: EdgeStackProps) {
    super(scope, id, props);

    const { thingGroupName, modelsBucket, telemetryBucket, inferenceRepo } = props;

    // --- IoT Thing Group ---
    // All robot edge devices belong to this group for batch operations
    const thingGroup = new iot.CfnThingGroup(this, 'RobotThingGroup', {
      thingGroupName,
      thingGroupProperties: {
        thingGroupDescription: 'Physical AI robot fleet — UR3 arms with Jetson/GPU edge compute',
        attributePayload: {
          attributes: {
            environment: props.environment,
            robotType: 'ur3',
            gripper: 'robotiq',
          },
        },
      },
    });

    // --- IoT Policy ---
    // Permissions for edge devices to connect, publish telemetry, and pull models
    const iotPolicy = new iot.CfnPolicy(this, 'RobotIotPolicy', {
      policyName: `physical-ai-${props.environment}-robot-policy`,
      policyDocument: {
        Version: '2012-10-17',
        Statement: [
          {
            Effect: 'Allow',
            Action: ['iot:Connect'],
            Resource: [`arn:aws:iot:${this.region}:${this.account}:client/\${iot:Connection.Thing.ThingName}`],
          },
          {
            Effect: 'Allow',
            Action: ['iot:Publish', 'iot:Receive'],
            Resource: [
              `arn:aws:iot:${this.region}:${this.account}:topic/physical-ai/telemetry/*`,
              `arn:aws:iot:${this.region}:${this.account}:topic/physical-ai/status/*`,
            ],
          },
          {
            Effect: 'Allow',
            Action: ['iot:Subscribe'],
            Resource: [
              `arn:aws:iot:${this.region}:${this.account}:topicfilter/physical-ai/commands/*`,
              `arn:aws:iot:${this.region}:${this.account}:topicfilter/physical-ai/models/*`,
            ],
          },
          {
            // Greengrass needs broader permissions for component deployment
            Effect: 'Allow',
            Action: [
              'greengrass:*',
              'iot:GetThingShadow',
              'iot:UpdateThingShadow',
            ],
            Resource: ['*'],
          },
        ],
      },
    });

    // --- IAM Role for Greengrass Token Exchange ---
    // Greengrass core devices assume this role to access AWS services
    const greengrassRole = new iam.Role(this, 'GreengrassTokenExchangeRole', {
      roleName: `physical-ai-${props.environment}-greengrass-role`,
      assumedBy: new iam.ServicePrincipal('credentials.iot.amazonaws.com'),
    });

    // Allow Greengrass devices to pull models from S3
    modelsBucket.grantRead(greengrassRole);

    // Allow Greengrass devices to push telemetry to S3
    telemetryBucket.grantWrite(greengrassRole);

    // Allow Greengrass to pull container images from ECR
    greengrassRole.addManagedPolicy(
      iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEC2ContainerRegistryReadOnly')
    );

    // Allow CloudWatch logging from edge devices
    greengrassRole.addToPolicy(new iam.PolicyStatement({
      actions: [
        'logs:CreateLogGroup',
        'logs:CreateLogStream',
        'logs:PutLogEvents',
      ],
      resources: ['*'],
    }));

    // --- Greengrass Component: Inference Node ---
    // This component runs the TensorRT inference + ROS 2 node on the edge device
    const inferenceComponent = new greengrassv2.CfnComponentVersion(this, 'InferenceComponent', {
      inlineRecipe: JSON.stringify({
        RecipeFormatVersion: '2020-01-25',
        ComponentName: `com.physicalai.${props.environment}.inference`,
        ComponentVersion: '1.0.0',
        ComponentDescription: 'TensorRT inference node for UR3 pick-and-place policy',
        ComponentPublisher: 'PhysicalAI',
        ComponentDependencies: {
          'aws.greengrass.DockerApplicationManager': {
            VersionRequirement: '>=2.0.0',
          },
          'aws.greengrass.TokenExchangeService': {
            VersionRequirement: '>=2.0.0',
          },
        },
        Manifests: [
          {
            Platform: { os: 'linux', architecture: 'aarch64' }, // Jetson
            Lifecycle: {
              run: {
                Script: 'docker run --rm --runtime nvidia --network host '
                  + '-v {artifacts:path}/model:/model '
                  + '{configuration:/containerUri} '
                  + '--model-path /model/policy.trt '
                  + '--ros-topic /ur3/joint_commands',
              },
            },
            Artifacts: [
              {
                Uri: `s3://${modelsBucket.bucketName}/latest/policy.trt`,
                Unarchive: 'NONE',
              },
            ],
          },
          {
            Platform: { os: 'linux', architecture: 'x86_64' }, // GPU PC
            Lifecycle: {
              run: {
                Script: 'docker run --rm --gpus all --network host '
                  + '-v {artifacts:path}/model:/model '
                  + '{configuration:/containerUri} '
                  + '--model-path /model/policy.trt '
                  + '--ros-topic /ur3/joint_commands',
              },
            },
            Artifacts: [
              {
                Uri: `s3://${modelsBucket.bucketName}/latest/policy.trt`,
                Unarchive: 'NONE',
              },
            ],
          },
        ],
        ComponentConfiguration: {
          DefaultConfiguration: {
            // Resolved CDK token → the real ECR image URI for this account/region.
            // (Previously a literal '${ECR_INFERENCE_REPO_URI}:latest' that never resolved.)
            containerUri: `${inferenceRepo.repositoryUri}:latest`,
          },
        },
      }),
    });

    // --- Greengrass Component: Telemetry Collector ---
    const telemetryComponent = new greengrassv2.CfnComponentVersion(this, 'TelemetryComponent', {
      inlineRecipe: JSON.stringify({
        RecipeFormatVersion: '2020-01-25',
        ComponentName: `com.physicalai.${props.environment}.telemetry`,
        ComponentVersion: '1.0.0',
        ComponentDescription: 'Collects grasp telemetry and uploads to S3 for retraining',
        ComponentPublisher: 'PhysicalAI',
        Manifests: [
          {
            Platform: { os: 'linux' },
            Lifecycle: {
              run: {
                Script: 'python3 {artifacts:path}/telemetry_collector.py '
                  + '--bucket ' + telemetryBucket.bucketName + ' '
                  + '--region ' + this.region,
              },
            },
            // The Lifecycle script references {artifacts:path}/telemetry_collector.py,
            // so the manifest MUST declare that artifact or Greengrass can't resolve it.
            // The collector is delivered to s3://<models bucket>/edge/telemetry_collector.py
            // by the edge deploy tooling (see docs/ROADMAP.md Feature 1).
            Artifacts: [
              {
                Uri: `s3://${modelsBucket.bucketName}/edge/telemetry_collector.py`,
                Unarchive: 'NONE',
              },
            ],
          },
        ],
      }),
    });

    // --- Outputs ---
    new cdk.CfnOutput(this, 'ThingGroupArn', { value: thingGroup.attrArn });
    new cdk.CfnOutput(this, 'GreengrassRoleArn', { value: greengrassRole.roleArn });
    new cdk.CfnOutput(this, 'DeviceSetupInstructions', {
      value: 'See docs/getting-started.md for Jetson/GPU PC provisioning steps',
    });
  }
}
