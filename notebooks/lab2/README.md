# Lab 2 — Isaac Sim Development Workstation

Deploy a GPU-powered remote desktop for visual development and debugging of RL robot
environments. Use this before Lab 4 (RL training on SageMaker) to validate that
your Isaac Lab environment works correctly.

---

## Quick Start

1. Read `BACKGROUND.md` — explains why visual debugging matters and what gets deployed
2. Open `Lab2_Isaac_Workstation.ipynb` and run top to bottom
3. Connect to the workstation via browser (NICE DCV, port 8443)
4. Develop and debug your Isaac Lab environment visually
5. **Stop the instance when done** (~$2.20/hr while running)

---

## Folder Structure

```
lab2/
├── README.md                          ← this file
├── BACKGROUND.md                      ← background for newcomers
└── Lab2_Isaac_Workstation.ipynb       ← single notebook (deploy + use + manage)
```

No container build needed for Lab 2. The Marketplace AMI comes pre-baked with
NVIDIA drivers, NICE DCV, and Isaac Sim. The bootstrap script handles Docker,
cache directories, and convenience scripts only.

---

## What Gets Deployed

- EC2 `g6e.4xlarge` (NVIDIA L40S 48 GB, ~$2.20/hr) — **required instance family**
  - The Marketplace AMI only supports `g6e` instances. `g5` and other families
    will fail with "instance configuration not supported".
- Ubuntu 24.04 + NVIDIA driver (pre-installed in AMI)
- NICE DCV remote desktop (browser access via HTTPS port 8443, pre-installed in AMI)
- Isaac Sim 5.1.0 in `~/IsaacSim` (pre-installed in AMI)
- Docker + NVIDIA Container Toolkit (installed by bootstrap)
- 512 GB encrypted EBS (persists across stop/start)

### What you'll see on first login (`ls ~`)
```
IsaacSim/                  ← Isaac Sim 5.1.0 (from AMI)
aws-physical-ai-toolchain/ ← toolchain code (bundled by CDK at deploy time)
isaac-cache/               ← shader/extension cache (speeds up subsequent boots)
run-isaac-sim-gui.sh       ← launches Isaac Sim GUI
run-isaac-lab.sh           ← launches isaac-lab Docker container (requires Lab 4 ECR image)
```

> **Note:** `~/isaac-env` mentioned in some older docs does not exist. Isaac Sim is
> at `~/IsaacSim`. Launch it with `./run-isaac-sim-gui.sh`.

---

## External Dependencies

This notebook is **not fully self-contained**. It requires:

| Dependency | Required? | Notes |
|-----------|-----------|-------|
| CDK (`cdk/`) | **Required** | TypeScript CDK app at `cdk/lib/workstation-stack.ts`. The notebook calls `npx cdk deploy` — CDK must be bootstrapped first. |
| Node.js ≥ 18 | **Required** | For CDK. Check: `node --version` |
| AWS Marketplace subscription | **Required** | Must subscribe to the NVIDIA Isaac Sim AMI **before** deploying. See below. |
| SageMaker execution role with CDK permissions | **Required** | See IAM Requirements below. |
| `physical-ai/isaac-lab` ECR image | Optional | Only needed for `run-isaac-lab.sh`. Built in Lab 4. |

### Marketplace Subscription (one-time, before first deploy)

The AMI is from the AWS Marketplace and requires accepting terms before EC2 can launch it.
Visit this URL and click **Subscribe → Accept Terms**:

https://aws.amazon.com/marketplace/pp/prodview-bl35herdyozhw

Forgetting this step causes: `"In order to use this AWS Marketplace product you need to
accept terms and subscribe"` error during CloudFormation stack creation.

### CDK Context Flags (required, not optional)

The workstation stack is **opt-in** in the CDK app. The deploy command must include:

```bash
--context workstation=true   # required — stack is not instantiated without this
--context mode=simple        # required — sets the CDK app mode
```

Without `workstation=true`, CDK synth succeeds but the stack doesn't exist, and
`cdk deploy PhysicalAi-dev-Workstation` fails with "No stacks match the name(s)".

---

## IAM Requirements

### Deploying role (e.g. SageMaker execution role)

The role running the notebook needs these permissions. Most are covered by standard
managed policies, but CDK bootstrap requires additional IAM and S3 permissions that
are **not** included in the default SageMaker execution role.

#### Standard managed policies (likely already attached)
- `AmazonEC2FullAccess` — launch the EC2 instance
- `AWSCloudFormationFullAccess` — create/update/delete the stack
- `AmazonS3FullAccess` or scoped S3 access — upload CDK assets to the bootstrap bucket
- `AmazonSSMFullAccess` — SSM send-command for bootstrap monitoring

#### Additional inline policy required for CDK bootstrap — `CDKBootstrapIAM`

CDK bootstrap creates IAM roles (`cdk-hnb659fds-*`) for CloudFormation execution.
The deploying role needs permission to manage those roles:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "CDKBootstrapIAM",
      "Effect": "Allow",
      "Action": [
        "iam:CreateRole",
        "iam:DeleteRole",
        "iam:AttachRolePolicy",
        "iam:DetachRolePolicy",
        "iam:PutRolePolicy",
        "iam:DeleteRolePolicy",
        "iam:GetRole",
        "iam:PassRole",
        "iam:TagRole"
      ],
      "Resource": "arn:aws:iam::*:role/cdk-hnb659fds-*"
    }
  ]
}
```

#### Additional inline policy required for CDK bootstrap — `CDKBootstrapS3`

CDK bootstrap creates and configures an S3 assets bucket. The deploying role needs
permission to set bucket policies on it:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "CDKBootstrapS3",
      "Effect": "Allow",
      "Action": [
        "s3:PutBucketPolicy",
        "s3:GetBucketPolicy",
        "s3:DeleteBucketPolicy",
        "s3:PutBucketVersioning",
        "s3:PutEncryptionConfiguration",
        "s3:PutBucketPublicAccessBlock",
        "s3:PutLifecycleConfiguration",
        "s3:PutBucketTagging",
        "s3:DeleteBucket"
      ],
      "Resource": "arn:aws:s3:::cdk-hnb659fds-assets-*"
    }
  ]
}
```

> These two policies are scoped tightly to CDK bootstrap resources only. They do not
> grant broad IAM or S3 admin access.

### Instance role (created by CDK, no action needed)

The workstation EC2 instance gets a role created automatically by the CDK stack with:
- `AmazonSSMManagedInstanceCore` — SSM Session Manager access (no SSH key needed)
- S3 read/write on `physical-ai-dev-*` buckets
- S3 read on `dcv-license-*` buckets (DCV license check)
- ECR pull on all repositories (to pull the isaac-lab training container)
- S3 read on the CDK assets bucket (to download the toolchain code bundle)

---

## Key Notes

> NICE DCV uses port **8443**. Disconnect VPN before connecting — most corporate
> VPNs block non-standard outbound ports.

> Isaac Sim first-launch takes **5–10 minutes** to compile shaders. If you see
> "Isaac Sim Full 5.1.0 is not responding", click **Wait** — it is still loading.
