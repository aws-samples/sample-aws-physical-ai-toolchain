# Lab 2: Isaac Sim Development Workstation

**Time:** 30 min setup + ongoing development sessions
**Cost:** ~$4.50/hr when running (stop when not in use)
**Goal:** Deploy a GPU-powered remote desktop for visual Isaac Lab environment development and debugging

---

## Why You Need This

You can't see what's happening inside a headless SageMaker training job. When your RL reward isn't improving, you need to *watch* the robot:

- Is the gripper missing the object by 2mm?
- Is the object falling through the table (physics bug)?
- Is the camera seeing what you think it's seeing?
- Is domain randomization too aggressive (objects flying off screen)?

The Isaac Sim workstation gives you a full visual desktop with GPU rendering, connected to the same environments you'll train headlessly on SageMaker. **Develop visually here, train at scale there.**

**The workflow:**
1. Deploy workstation (one time, ~5 min)
2. Start instance when you need to develop
3. Connect via web browser (NICE DCV remote desktop)
4. Iterate on your RL environment visually — see the robot, tweak rewards, fix bugs
5. Once it works visually → launch headless training on SageMaker (Lab 3)
6. Stop instance when done

---

## Cost Breakdown

| Resource | Cost | When |
|----------|------|------|
| g6e.4xlarge (L40S GPU) | $4.53/hr | Only while instance is running |
| 512 GB gp3 EBS | ~$40/month | Always (stores Isaac Sim + your work) |
| Elastic IP | $3.60/month | While allocated (free when attached to running instance) |

**Typical monthly cost:**
- Heavy development (8 hrs/day, 5 days/week): ~$720/month
- Moderate development (4 hrs/day, 3 days/week): ~$216/month
- Occasional debugging (2 hrs/week): ~$36/month

**Compared to alternatives:**
- Local GPU workstation (RTX 4090): $2,500+ upfront, limited to one developer
- NVIDIA Omniverse Cloud: $1000+/month per seat
- This workstation: pay only for hours used, accessible from any laptop via browser

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  EC2 g6e.4xlarge (NVIDIA L40S GPU)                          │
│                                                             │
│  ┌─────────────────┐  ┌──────────────────────────────────┐ │
│  │  NICE DCV       │  │  Isaac Sim + Isaac Lab            │ │
│  │  Remote Desktop │  │  - Visual scene editor            │ │
│  │  (port 8443)    │  │  - RL environment preview         │ │
│  │                 │  │  - Physics debugger               │ │
│  └─────────────────┘  │  - Domain randomization viewer    │ │
│                        └──────────────────────────────────┘ │
│  ┌─────────────────┐  ┌──────────────────────────────────┐ │
│  │  ROS2 Jazzy     │  │  Docker + NVIDIA Container Toolkit│ │
│  │  (robot comms)  │  │  (test training containers)       │ │
│  └─────────────────┘  └──────────────────────────────────┘ │
│                                                             │
│  512 GB gp3 SSD | S3 access | ECR pull access              │
└─────────────────────────────────────────────────────────────┘
         │
         │ NICE DCV (port 8443, encrypted)
         ▼
┌─────────────────┐
│  Your Laptop    │
│  (web browser)  │
└─────────────────┘
```

---

## Step 1: Deploy the Workstation

```bash
cd aws-physical-ai-toolchain/cdk

# Get your public IP
MY_IP=$(curl -s ifconfig.me)
echo "Your IP: $MY_IP"

# Deploy (takes ~5 minutes)
npx cdk deploy PhysicalAi-dev-Workstation \
  --context mode=simple \
  --context allowedCidr="$MY_IP/32"
```

The stack outputs will show:
- **WorkstationIP** — the Elastic IP address
- **DCVWebURL** — `https://<IP>:8443` to connect via browser
- **SSMConnect** — connect via Session Manager (no SSH key needed)

---

## Step 2: Set the DCV Password

```bash
# Get the instance ID from the stack output, then set password
INSTANCE_ID=$(aws cloudformation describe-stacks \
  --stack-name PhysicalAi-dev-Workstation \
  --query 'Stacks[0].Outputs[?OutputKey==`SSMConnect`].OutputValue' \
  --output text | grep -oP 'i-\w+')

# Set password for DCV login (replace YOUR_PASSWORD)
aws ssm send-command \
  --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["echo ubuntu:YOUR_PASSWORD | chpasswd"]'
```

---

## Step 3: Connect via Browser

1. Open `https://<WorkstationIP>:8443` in your browser
2. Accept the self-signed certificate warning
3. Login: username `ubuntu`, password you just set
4. You'll see an Ubuntu desktop with GPU acceleration

---

## Step 4: Launch Isaac Sim

On the workstation desktop, open a terminal:

```bash
# Isaac Sim is pre-installed via the container or pip
# Option A: Run Isaac Sim standalone
~/.local/share/ov/pkg/isaac-sim-4.5.0/isaac-sim.sh

# Option B: Launch Isaac Lab with our UR3 environment (visual mode)
cd /workspace/isaaclab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Velocity-Flat-Anymal-D-v0 \
  --num_envs=16 \
  --max_iterations=10
```

You'll see the simulated robots training in real-time with full rendering.

---

## Step 5: Develop Your RL Environment

This is where you iterate:

```bash
# Clone your repo on the workstation
git clone git@ssh.gitlab.aws.dev:devris/aws-physical-ai-toolchain.git
cd aws-physical-ai-toolchain

# Run the UR3 pick-and-place environment visually
./isaaclab.sh -p training/envs/pick_and_place_ur3.py --num_envs=8

# Watch the robot attempt the task
# Tweak rewards, observation space, action space
# When it looks right → launch headless on SageMaker (Lab 3)
```

**What to look for:**
- Robot reaching toward the object (approach reward working)
- Gripper closing at the right time (grasp reward working)
- Object being lifted cleanly (success reward working)
- No physics glitches (objects clipping through surfaces)
- Domain randomization looking reasonable (not too wild)

---

## Start/Stop the Workstation

**Stop when done (saves money):**
```bash
INSTANCE_ID=<your-instance-id>
aws ec2 stop-instances --instance-ids $INSTANCE_ID
```

**Start when you need it again:**
```bash
aws ec2 start-instances --instance-ids $INSTANCE_ID
# Wait ~60s for boot, then reconnect via DCV
```

Your work is preserved on the EBS volume — stopping only halts the compute charges.

---

## Testing Training Containers Locally

The workstation has Docker + NVIDIA Container Toolkit, so you can test your SageMaker training containers without waiting for SageMaker provisioning:

```bash
# Pull the Isaac Lab training container from ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 802782083985.dkr.ecr.us-east-1.amazonaws.com
docker pull 802782083985.dkr.ecr.us-east-1.amazonaws.com/physical-ai/isaac-lab:latest

# Run it like SageMaker would (simulating the training invocation)
docker run --gpus all \
  -v /tmp/test-output:/opt/ml/model \
  -v /tmp/test-config:/opt/ml/input/config \
  802782083985.dkr.ecr.us-east-1.amazonaws.com/physical-ai/isaac-lab:latest \
  train

# Iterate on container changes in seconds instead of waiting for CodeBuild + SageMaker
```

This is the fastest iteration loop for container issues (like the entrypoint fix we debugged).

---

## ✅ Lab 2 Checkpoint

You've completed Lab 2 if:
- [ ] Workstation is deployed and accessible via DCV
- [ ] You can launch Isaac Sim and see rendered environments
- [ ] You can run the training environment visually with a small number of envs
- [ ] You understand the start/stop workflow to manage costs
- [ ] You can pull and test training containers locally on the workstation

---

## Tips for Cost Management

1. **Always stop when you walk away.** Set a calendar reminder or use AWS Instance Scheduler.
2. **Use Spot instances for non-critical work.** Modify the CDK stack to use Spot — saves ~70% but can be interrupted.
3. **Right-size the instance.** g6e.4xlarge (1× L40S) is sufficient for environment development. Only upgrade to g6e.8xlarge if you need to run 4096+ envs visually.
4. **Delete when the project is done.** `cdk destroy PhysicalAi-dev-Workstation` removes everything.

---

**Previous:** [← Lab 1: Train from Demonstrations](lab-1-train-groot.md)
**Next:** [Lab 3: RL Refinement in Simulation →](lab-3-rl-refinement.md)
