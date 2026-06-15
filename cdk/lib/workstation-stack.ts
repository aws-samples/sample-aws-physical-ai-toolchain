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

    // For dev environments, allow all IPs since DCV requires password authentication.
    // For production, restrict to corporate CIDR via the allowedCidr context parameter.
    // Note: VPN often blocks port 8443 outbound — users typically connect off-VPN.
    const allowedCidr = props.allowedCidr || '0.0.0.0/0';
    const instanceType = props.instanceType || 'g5.4xlarge';

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

    // UserData: Full bootstrap based on proven pattern from
    // github.com/aws-samples/sample-physical-ai-scaffolding-kit
    const userData = ec2.UserData.forLinux();
    userData.addCommands(
      '#!/bin/bash',
      'set -e',
      'exec > /var/log/workstation-bootstrap.log 2>&1',
      'export DEBIAN_FRONTEND=noninteractive',
      '',
      '# Wait for apt locks to clear',
      'while fuser /var/lib/dpkg/lock >/dev/null 2>&1; do sleep 3; done',
      'while fuser /var/lib/apt/lists/lock >/dev/null 2>&1; do sleep 3; done',
      '',
      'echo "=== Step 1: Base packages ==="',
      'apt-get update -yq',
      'apt-get install -yq ca-certificates curl wget gnupg lsb-release jq unzip',
      '',
      'echo "=== Step 2: NVIDIA driver ==="',
      'apt-get install -yq ubuntu-drivers-common',
      'ubuntu-drivers autoinstall',
      '',
      'echo "=== Step 3: Desktop + GDM ==="',
      'apt-get install -yq ubuntu-desktop gdm3 dbus-x11',
      'sed -i "s/^#\\(WaylandEnable=false\\)/\\1/" /etc/gdm3/custom.conf || true',
      '',
      'echo "=== Step 4: Install NICE DCV ==="',
      'cd /tmp',
      'wget -q "https://d1uj6qtbmh3dt5.cloudfront.net/2024.0/Servers/nice-dcv-2024.0-19030-ubuntu2204-x86_64.tgz" -O /tmp/dcv.tgz',
      'tar -xzf /tmp/dcv.tgz -C /tmp',
      'cd /tmp/nice-dcv-2024.0-19030-ubuntu2204-x86_64',
      'apt-get install -yq ./nice-dcv-server_*.deb ./nice-dcv-web-viewer_*.deb ./nice-xdcv_*.deb || apt-get install -yf',
      'usermod -aG video dcv || true',
      'systemctl enable dcvserver',
      'systemctl restart dcvserver',
      '',
      'echo "=== Step 5: DCV auto-session ==="',
      'cat > /usr/local/bin/auto-create-dcv.sh << \'SCRIPT\'',
      '#!/bin/bash',
      'sleep 5',
      'until systemctl is-active --quiet dcvserver; do sleep 3; done',
      'if ! dcv list-sessions | grep -q "^Session:"; then',
      '  dcv create-session --type virtual --owner ubuntu --name "Isaac Sim" main',
      'fi',
      'SCRIPT',
      'chmod +x /usr/local/bin/auto-create-dcv.sh',
      'cat > /etc/systemd/system/auto-dcv.service << \'SVC\'',
      '[Unit]',
      'Description=Auto-create DCV session',
      'After=dcvserver.service',
      '[Service]',
      'Type=oneshot',
      'ExecStart=/usr/local/bin/auto-create-dcv.sh',
      'RemainAfterExit=yes',
      '[Install]',
      'WantedBy=multi-user.target',
      'SVC',
      'systemctl daemon-reload',
      'systemctl enable auto-dcv.service',
      '',
      'echo "=== Step 6: Docker + NVIDIA Container Toolkit ==="',
      'curl -fsSL https://get.docker.com | sh',
      'systemctl enable docker && systemctl start docker',
      'usermod -aG docker ubuntu || true',
      'curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg',
      'curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null',
      'apt-get update -yq && apt-get install -yq nvidia-container-toolkit',
      'systemctl restart docker',
      '',
      'echo "=== Step 7: Set ubuntu password ==="',
      'echo "ubuntu:physical-ai-2026" | chpasswd',
      '',
      'echo "=== Step 8: AWS CLI v2 ==="',
      'curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip',
      'unzip -q /tmp/awscliv2.zip -d /tmp',
      '/tmp/aws/install --update',
      '',
      'echo "Workstation bootstrap complete" > /var/log/workstation-bootstrap.summary',
      'echo "DCV URL: https://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8443"',
      'echo "Username: ubuntu  Password: physical-ai-2026"',
      '',
      '# Reboot to finalize NVIDIA driver + desktop',
      'shutdown -r +1 "Rebooting to finalize workstation setup"',
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

    // Outputs
    new cdk.CfnOutput(this, 'WorkstationInstanceId', {
      value: instance.instanceId,
      description: 'Instance ID (use for start/stop and SSM)',
    });

    new cdk.CfnOutput(this, 'DCVWebURL', {
      value: `Connect via: https://<PUBLIC_IP>:8443 (get IP from EC2 console or: aws ec2 describe-instances --instance-ids ${instance.instanceId} --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)`,
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
      value: `~$4.53/hr (${instanceType} on-demand). STOP instance when not in use! Start/stop via console or: aws ec2 stop-instances / start-instances`,
      description: 'Estimated hourly cost — only runs when you need it',
    });
  }
}
