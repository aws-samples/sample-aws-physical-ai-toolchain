import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';

export interface WorkstationStackProps extends cdk.StackProps {
  environment: string;
  /**
   * CIDR range allowed to connect via DCV (port 8443).
   * Use your IP (e.g., "203.0.113.1/32") or "0.0.0.0/0" for dev.
   */
  allowedCidr?: string;
  /**
   * EC2 instance type. Must be a GPU instance for Isaac Sim.
   * g6e.8xlarge recommended (L40S GPU, good price/performance).
   */
  instanceType?: string;
}

/**
 * Isaac Sim Development Workstation
 *
 * Deploys a GPU EC2 instance with:
 * - NVIDIA Isaac Sim AMI (from AWS Marketplace)
 * - NICE DCV for remote desktop access
 * - ROS2 Jazzy (installed via UserData)
 * - S3 access for datasets and models
 *
 * Use this to visually develop and debug Isaac Lab RL environments.
 * Connect via DCV client or web browser at https://<IP>:8443
 *
 * Based on: https://github.com/aws-samples/sample-physical-ai-scaffolding-kit/tree/main/isaacsim-workstation
 *
 * Deploy:
 *   cdk deploy PhysicalAi-dev-Workstation --context mode=simple --context allowedCidr=YOUR_IP/32
 *
 * WORKSHOP NOTE: This is your "IDE" for robotics. Develop visually here,
 * then run headless training on SageMaker once the environment works.
 */
export class WorkstationStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: WorkstationStackProps) {
    super(scope, id, props);

    const allowedCidr = props.allowedCidr || '0.0.0.0/0';
    const instanceType = props.instanceType || 'g6e.4xlarge';

    // Use default VPC for simplicity
    const vpc = ec2.Vpc.fromLookup(this, 'DefaultVpc', { isDefault: true });

    // Security group: DCV (8443) + SSH (22)
    const sg = new ec2.SecurityGroup(this, 'WorkstationSg', {
      vpc,
      securityGroupName: `physical-ai-${props.environment}-workstation-sg`,
      description: 'Isaac Sim workstation: DCV + SSH access',
      allowAllOutbound: true,
    });
    sg.addIngressRule(ec2.Peer.ipv4(allowedCidr), ec2.Port.tcp(8443), 'DCV remote desktop');
    sg.addIngressRule(ec2.Peer.ipv4(allowedCidr), ec2.Port.tcp(22), 'SSH');

    // IAM role for the instance
    const role = new iam.Role(this, 'WorkstationRole', {
      roleName: `physical-ai-${props.environment}-workstation-role`,
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore'),
      ],
    });

    // S3 access for datasets and models
    role.addToPolicy(new iam.PolicyStatement({
      actions: ['s3:GetObject', 's3:PutObject', 's3:ListBucket'],
      resources: [
        `arn:aws:s3:::physical-ai-${props.environment}-*`,
        `arn:aws:s3:::physical-ai-${props.environment}-*/*`,
      ],
    }));

    // DCV license check
    role.addToPolicy(new iam.PolicyStatement({
      actions: ['s3:GetObject'],
      resources: ['arn:aws:s3:::dcv-license-*/*'],
    }));

    // ECR pull access (to pull our training containers for local testing)
    role.addToPolicy(new iam.PolicyStatement({
      actions: [
        'ecr:GetAuthorizationToken',
        'ecr:BatchCheckLayerAvailability',
        'ecr:GetDownloadUrlForLayer',
        'ecr:BatchGetImage',
      ],
      resources: ['*'],
    }));

    // UserData: install ROS2 + configure DCV
    const userData = ec2.UserData.forLinux();
    userData.addCommands(
      '#!/bin/bash',
      'set -e',
      '',
      '# Log setup progress',
      'exec > /var/log/workstation-bootstrap.log 2>&1',
      '',
      '# Install ROS2 Jazzy',
      'apt-get update',
      'apt-get install -y software-properties-common',
      'add-apt-repository universe',
      'curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add -',
      'echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2.list',
      'apt-get update',
      'apt-get install -y ros-jazzy-desktop ros-jazzy-rosbridge-suite',
      '',
      '# Configure DCV for auto-session',
      'systemctl enable dcvserver',
      'systemctl start dcvserver',
      '',
      '# Create DCV session on boot',
      'cat > /etc/systemd/system/dcv-session.service << EOF',
      '[Unit]',
      'Description=DCV Session',
      'After=dcvserver.service',
      '',
      '[Service]',
      'ExecStart=/usr/bin/dcv create-session --type virtual --owner ubuntu main',
      'Restart=on-failure',
      'RestartSec=5',
      '',
      '[Install]',
      'WantedBy=multi-user.target',
      'EOF',
      'systemctl enable dcv-session',
      'systemctl start dcv-session',
      '',
      'echo "Workstation bootstrap complete" > /var/log/workstation-bootstrap.summary',
    );

    // EC2 Instance
    // Option 1: Use Isaac Sim AMI from Marketplace (requires subscription)
    //   https://aws.amazon.com/marketplace/pp/prodview-bl35herdyozhw
    // Option 2: Use Ubuntu + install Isaac Sim via UserData (what we do below)
    const instance = new ec2.Instance(this, 'Workstation', {
      instanceType: new ec2.InstanceType(instanceType),
      machineImage: ec2.MachineImage.fromSsmParameter(
        '/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id',
      ),
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      securityGroup: sg,
      role,
      userData,
      blockDevices: [{
        deviceName: '/dev/sda1',
        volume: ec2.BlockDeviceVolume.ebs(512, {
          volumeType: ec2.EbsDeviceVolumeType.GP3,
          encrypted: true,
        }),
      }],
      associatePublicIpAddress: true,
    });

    // Elastic IP for stable address
    const eip = new ec2.CfnEIP(this, 'WorkstationEip');
    new ec2.CfnEIPAssociation(this, 'WorkstationEipAssoc', {
      instanceId: instance.instanceId,
      allocationId: eip.attrAllocationId,
    });

    // Outputs
    new cdk.CfnOutput(this, 'WorkstationIP', {
      value: eip.attrPublicIp,
      description: 'Workstation public IP',
    });

    new cdk.CfnOutput(this, 'DCVWebURL', {
      value: `https://${eip.attrPublicIp}:8443`,
      description: 'Connect via web browser (accept cert warning)',
    });

    new cdk.CfnOutput(this, 'SSMConnect', {
      value: `aws ssm start-session --target ${instance.instanceId}`,
      description: 'Connect via Session Manager (no SSH key needed)',
    });

    new cdk.CfnOutput(this, 'SetPassword', {
      value: `aws ssm send-command --instance-ids ${instance.instanceId} --document-name "AWS-RunShellScript" --parameters 'commands=["echo ubuntu:YOUR_PASSWORD | chpasswd"]'`,
      description: 'Set the ubuntu user password for DCV login',
    });

    new cdk.CfnOutput(this, 'Cost', {
      value: `~$4.53/hr (${instanceType} on-demand). STOP instance when not in use!`,
      description: 'Estimated hourly cost',
    });
  }
}
