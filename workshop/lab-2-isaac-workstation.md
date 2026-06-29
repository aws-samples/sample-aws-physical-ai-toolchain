# Lab 2: Isaac Sim Development Workstation

**Goal:** Deploy a GPU-powered remote desktop for visual Isaac Lab environment development and debugging
**Time:** 30 min setup + ongoing development sessions
**Cost:** ~$3.00/hr + $40/month EBS when running (stop when not in use)

---

## 🏃 Quick Runbook (do this in order)

> Follow these steps top-to-bottom. Each step says **what to run** and **how you know it worked**. Full detail for each is in the numbered sections below.

| # | Action | Command (summary) | ✅ Success check |
|---|--------|-------------------|-----------------|
| 0 | Subscribe to the Isaac Sim AMI (once per account) | Marketplace console — see *Prerequisite* | Subscription shows **Active** |
| 1 | Set your region in `config.json` | `pai config set aws.region us-west-2` | matches the region you'll deploy in |
| 2 | Deploy the workstation | `pai deploy workstation` | `✅ CREATE_COMPLETE`, prints instance ID + IP |
| 3 | Get IP and password | `pai workstation ip` / `pai workstation password` | prints IP and sets password |
| 4 | Connect via browser | open `https://<IP>:8443` | DCV login → Ubuntu desktop renders |
| 5 | Launch Isaac Sim (visual) | `~/run-isaac-sim-gui.sh` | 3D viewport opens |
| 6 | Visual RL training in the container | `~/run-isaac-lab.sh` → `./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Velocity-Flat-Anymal-D-v0` | render window + training logs steps/s |
| 7 | (Optional) Closed-loop policy eval | inside the same container: `eval_policy_server.py` + `eval_sim_client.py` (two shells) | prints `success_rate` JSON (**unvalidated on GPU**) |
| 8 | **Stop the instance** | `pai workstation stop` | state → `stopped` (billing halts) |

**Before you start, confirm:**
- [ ] AWS credentials active for the **test account** (`aws sts get-caller-identity`)
- [ ] `config.json` `aws.region` is the region you intend (deploy reads it — Step 1)
- [ ] You completed the Marketplace **subscription** (Step 0 / *Prerequisite* below)
- [ ] Foundation stack already deployed (so the `isaac-lab` image is in your ECR) — needed only for Step 5

> 💸 **Cost reminder:** this instance bills ~$3.00/hr while running. Do Step 6 the moment you walk away.

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
| g6e.4xlarge (L40S GPU, 48 GB VRAM) | ~$3.00/hr | Only while instance is running |
| 512 GB gp3 EBS | ~$40/month | Always (stores Isaac Sim + your work) |

**Typical monthly cost:**
- Heavy development (8 hrs/day, 5 days/week): ~$520/month (160 hrs × $3.00 + $40 EBS)
- Moderate development (4 hrs/day, 3 days/week): ~$184/month (48 hrs × $3.00 + $40 EBS)
- Occasional debugging (2 hrs/week): ~$64/month (8 hrs × $3.00 + $40 EBS)

> **Why g6e.4xlarge?** This is the instance NVIDIA recommends for the Isaac Sim Marketplace AMI (1× L40S, 48 GB VRAM). You can override it in `config.json` (`workstation.instanceType`) or with `INSTANCE_TYPE=… ./deploy-workstation.sh`.

**Compared to alternatives:**
- Local GPU workstation (RTX 4090): $2,500+ upfront, limited to one developer
- NVIDIA Omniverse Cloud: $1000+/month per seat
- This workstation: pay only for hours used, accessible from any laptop via browser

---

## Architecture

> TODO: Insert workstation architecture diagram here (showing EC2 g6e.4xlarge with DCV, Isaac Sim, Docker connecting to laptop browser)

```
┌─────────────────────────────────────────────────────────────┐
│  EC2 g6e.4xlarge (NVIDIA L40S GPU, 48GB VRAM)              │
│  NVIDIA Isaac Sim Marketplace AMI (driver + DCV pre-baked) │
│                                                             │
│  ┌─────────────────┐  ┌──────────────────────────────────┐ │
│  │  NICE DCV       │  │  Isaac Sim + Isaac Lab            │ │
│  │  Remote Desktop │  │  - Visual scene editor            │ │
│  │  (port 8443)    │  │  - RL environment preview         │ │
│  │                 │  │  - Physics debugger               │ │
│  └─────────────────┘  │  - Domain randomization viewer    │ │
│                        └──────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Docker + NVIDIA Container Toolkit                       ││
│  │  (test training containers locally — same as SageMaker)  ││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
│  512 GB gp3 SSD | S3 access | ECR pull access               │
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

## Prerequisite: Subscribe to the Isaac Sim Marketplace AMI (one time)

This workstation boots the **NVIDIA Isaac Sim AMI** from AWS Marketplace, which ships with the NVIDIA driver, NICE DCV, Docker, and Isaac Sim **pre-installed** (no fragile from-scratch bootstrap). You must subscribe to it **once per account** before deploying, or the EC2 launch fails with a subscription/opt-in error:

1. Open the listing: **https://aws.amazon.com/marketplace/pp/prodview-bl35herdyozhw**
2. Click **Continue to Subscribe** → **Accept Terms** (the AMI itself is free; you pay only for the EC2 instance + EBS).
3. Wait for the subscription to show **Active** (usually < 1 min).

> **Region note:** The region→AMI map lives in **`config.json`** (`workstation.amiMapping`). Marketplace AMI IDs change with each Isaac Sim release — if your region is missing or the deploy reports an invalid AMI, look up the current AMI ID for your region from the listing's **Launch** tab and update `config.json`.

---

## Step 1: Deploy the Workstation

All workstation settings (instance type, region→AMI map, EBS size, DCV port) now live in **`config.json`** at the repo root. The deploy reads them automatically and auto-detects your public IP for the security group.

```bash
# Deploy with auto IP detection and AZ retry on GPU capacity errors
pai deploy workstation

# To override the instance type
pai deploy workstation --instance-type g5.2xlarge

# To use a specific IP (e.g., if behind NAT or VPN)
pai deploy workstation --allowed-cidr "203.0.113.5/32"
```

The CLI auto-detects your public IP (via `ifconfig.me`) and will retry across Availability Zones if GPU capacity is exhausted in one AZ. If DCV won't connect after deployment, your browser may be behind a different IP — re-deploy with that IP in `--allowed-cidr`.

The command outputs:
- **WorkstationInstanceId** — the EC2 instance ID (use it for start/stop)
- **DCVWebURL** — how to connect via browser. The instance uses an **auto-assigned public IP** (not an Elastic IP), so this output gives you the command to fetch the current IP: `pai workstation ip`. ⚠️ The public IP **changes every time you stop/start** — re-fetch it after each start.
- **Cost** — ~$3.00/hr (on-demand g6e.4xlarge in us-west-2) + EBS

> ⚠️ **Note on the public IP:** because it's not an Elastic IP, your `allowedCidr` security-group rule is unaffected by stop/start (that's keyed to *your* IP), but the DCV URL you bookmark will change. Always run `pai workstation ip` after starting the instance.

<details>
<summary>Under the hood (raw commands)</summary>

The `pai deploy workstation` command wraps:

```bash
cd aws-physical-ai-toolchain/cdk

# Deploy with the wrapper script
./deploy-workstation.sh

# Or use CDK directly
MY_IP=$(curl -s ifconfig.me)
npx cdk deploy PhysicalAi-dev-Workstation \
  --context mode=simple \
  --context workstation=true \
  --context allowedCidr="$MY_IP/32"
```

</details>

---

## Step 2: Connect via Browser

The workstation auto-configures everything on first boot (~15 min). Once ready:

```bash
# Get the current public IP
pai workstation ip

# Set the DCV password (if not already set)
pai workstation password

# Or print connection details
pai workstation connect
```

Then:

1. Open `https://<IP>:8443` in your browser (use IP from `pai workstation ip`)
2. Accept the self-signed certificate warning
3. Login: username `ubuntu`, password from `pai workstation password` (default: `pai-lab1`)
4. You'll see an Ubuntu desktop with GPU acceleration

> **Note:** DCV requires direct internet access (port 8443). If you're on a corporate VPN that blocks non-standard ports, disconnect VPN to access DCV.

<details>
<summary>Under the hood (raw commands)</summary>

```bash
# Get IP
INSTANCE_ID=<your-instance-id>
aws ec2 describe-instances --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text

# Set password via SSM
aws ssm send-command \
  --instance-ids $INSTANCE_ID \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["echo ubuntu:YOUR_PASSWORD | sudo chpasswd"]'

# Connect via SSM (alternative)
aws ssm start-session --target $INSTANCE_ID
```

</details>

---

## Step 3: Launch Isaac Sim (Visual Path)

Because the workstation now boots the **NVIDIA Isaac Sim Marketplace AMI**, Isaac Sim is **pre-installed by NVIDIA** — you no longer rely on a fragile from-scratch install. The AMI installs it to **`/opt/IsaacSim`** (also mirrored under `~/IsaacSim`), and the launcher is **not on your `PATH`**, so call it by absolute path from the DCV desktop terminal:

```bash
# Isaac Sim ships with the AMI at /opt/IsaacSim. Launch the GUI:
/opt/IsaacSim/isaac-sim.sh

# Convenience: the bootstrap also dropped ~/run-isaac-sim-gui.sh which calls the above.
~/run-isaac-sim-gui.sh
```

> **Note:** The install path (`/opt/IsaacSim`) is defined by the AMI and verified on the current Isaac Sim Marketplace release. If a future AMI moves it, find the launcher with `ls /opt/IsaacSim/isaac-sim.sh ~/IsaacSim/isaac-sim.sh`. For RL training, use the **`isaac-lab` container** (Step 4) — the same image SageMaker runs.

You'll see the full Isaac Sim visual editor — 3D viewport, content browser with robots and environments, scene tree.

---

## Step 4: Develop Your RL Environment

This is where you iterate on the UR3 pick-and-place environment. The toolchain code
is already on the workstation — the deploy bundles your working tree as an S3 asset
and the bootstrap unzips it to `/home/ubuntu/aws-physical-ai-toolchain`:

```bash
cd ~/aws-physical-ai-toolchain
```

> (The public GitHub repo isn't released yet, so there's no `git clone` — once it's
> public you can pass `--context repoUrl=<url>` to clone instead.)

### One environment for everything: the `isaac-lab` container

All GPU work on this workstation — visual training, interactive iteration, and policy
eval — runs **inside the `isaac-lab` Docker container**. This is the *same image*
SageMaker and AWS Batch run in Lab 4 (Isaac Sim 4.5.0 + Isaac Lab v2.1.0), so there's
**one consistent environment** end to end: what you see render here is exactly what
trains at scale. The container also carries the `omni.isaac.lab.*` packages this repo's
envs and eval scripts import. (The Marketplace AMI's *native* Isaac Sim is used only for
the standalone GUI in Step 3; RL goes through the container.)

The bootstrap pre-pulls the image and drops a launcher that wires up GPU + GUI
passthrough to your DCV desktop:

```bash
# Launch the container with a render window on the DCV desktop, land in a shell:
~/run-isaac-lab.sh
# (prompt becomes /workspace/isaaclab# — you are now inside the container)
```

> ✅ **GUI passthrough validated on this AMI** (g6e.4xlarge / L40S, driver 580): the
> container's RTX/Vulkan renderer initializes against the DCV display and an Anymal RL
> task trains to completion. A non-fatal `Warp CUDA error: cuDeviceGetUuid` may print on
> boot (container CUDA vs. host driver) — the sim runs through it. The **first** GUI boot
> is slow (~5 min: extension sync + shader compile); the launcher mounts persistent cache
> dirs so later boots are much faster.

Inside the container, train and watch it render — or run headless for max throughput:

```bash
# Visual training (render window opens on the DCV desktop):
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-Velocity-Flat-Anymal-D-v0 --num_envs 4096

# Headless (faster, no window):
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-Velocity-Flat-Anymal-D-v0 --num_envs 4096 --headless

# Play back / evaluate a checkpoint (visual):
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
  --task Isaac-Velocity-Flat-Anymal-D-v0 --num_envs 32 \
  --checkpoint logs/skrl/<run-dir>/checkpoints/agent_<N>.pt
```

Your repo working tree is mounted at `/workspace/toolchain` inside the container, so you
can edit `training/envs/pick_and_place_ur3.py` on the host (or in the DCV desktop's
editor) and re-run immediately — no rebuild.

**What to look for in the render window:**
- Robots tracking the commanded velocity (locomotion reward working)
- Stable gait, no flipping or limb clipping through the floor
- For the UR3 pick-place env: arm reaching, gripper closing on contact, clean lift

### The UR3 pick-and-place env (registered, GPU-untested)

`training/envs/pick_and_place_ur3.py` imports `omni.isaac.lab.*` — the namespace the
container provides — so it loads in this environment. **But env *instantiation* on the
GPU has not been validated**; the verified RL path here is the built-in Anymal task.
Treat UR3 as the thing you're bringing up, not a known-good baseline.

---

## Start/Stop the Workstation

**Stop when done (saves money):**
```bash
pai workstation stop
```

**Start when you need it again:**
```bash
pai workstation start

# Wait ~60s for boot, then get the new public IP and reconnect via DCV:
pai workstation ip
```

**Check current status:**
```bash
pai workstation status
```

> ⚠️ The instance has an **auto-assigned public IP, not an Elastic IP** — it changes on every stop/start. Re-run `pai workstation ip` after each start to get the current `https://<IP>:8443` URL.

Your work is preserved on the EBS volume — stopping only halts the compute charges.

<details>
<summary>Under the hood (raw commands)</summary>

```bash
INSTANCE_ID=<your-instance-id>

# Stop
aws ec2 stop-instances --instance-ids $INSTANCE_ID

# Start
aws ec2 start-instances --instance-ids $INSTANCE_ID

# Get IP
aws ec2 describe-instances --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
```

</details>

---

## Running the Container Exactly as SageMaker Does (parity check)

Step 4 launches the `isaac-lab` container *interactively* with GUI passthrough so you
can watch training render. This section runs the **same image** through its `train`
entrypoint **non-interactively** — exactly how SageMaker/Batch invoke it in Lab 4 — so
you can reproduce a job's behavior locally before launching at scale. Same container,
different entrypoint; no GUI.

> The image is `physical-ai/isaac-lab:latest`, built by **CodeBuild into your ECR when
> you deploy the Foundation stack**. This works only **after** Foundation is deployed in
> *this* account/region and the build has finished. If not, the `docker pull` returns
> `not found` — finish Lab 1 (or deploy Foundation) first. (The workstation bootstrap
> pre-pulls it best-effort and logs a non-fatal "image not in ECR yet" if it's missing.)

Pull the CodeBuild-built image from *your* ECR and run it as SageMaker would, without
waiting for SageMaker provisioning. The registry is derived from your own
account/region, so nothing is hardcoded:

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region)
ECR_REGISTRY="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
IMAGE="$ECR_REGISTRY/physical-ai/isaac-lab:latest"

# Pull the Isaac Lab training container from ECR (built for you by CodeBuild)
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY"
docker pull "$IMAGE"

# Run it like SageMaker would (simulating the training invocation)
docker run --gpus all \
  -v /tmp/test-output:/opt/ml/model \
  -v /tmp/test-config:/opt/ml/input/config \
  "$IMAGE" \
  train
```

This is the fastest iteration loop for container issues. If you change the
Dockerfile, you can rebuild in the cloud (`aws codebuild start-build
--project-name physical-ai-isaac-lab-build`) or, on this x86 workstation, build
locally and push to ECR yourself.

---

## Closed-Loop Policy Evaluation (Lab 4 Step 4 — runs here)

**Lab 4's closed-loop evaluator runs on this workstation** because it needs the L40S
GPU. It splits the policy and simulator into two processes — a **policy server** and a
**sim client** that drives Isaac Lab — mirroring how the policy is served on the robot
(Lab 5). Both run **inside the `isaac-lab` container** (launch with `~/run-isaac-lab.sh`,
open a second shell with `sudo docker exec -it isaac-lab bash`); the repo is mounted at
`/workspace/toolchain`.

The full walkthrough — scriptifying the checkpoint and the two-shell commands — lives in
**[Lab 4 → Step 4](lab-4-rl-refinement.md#step-4-evaluate-the-trained-policy-closed-loop)**.
(Closed-loop eval is **unvalidated on GPU** — see that step's note.)

---

## ✅ Lab 2 Checkpoint

You've completed Lab 2 if:
- [ ] Workstation is deployed and accessible via DCV
- [ ] You can launch Isaac Sim and see rendered environments
- [ ] You can run the training environment visually with a small number of envs
- [ ] You understand the start/stop workflow to manage costs
- [ ] You can pull and test training containers locally on the workstation
- [ ] (Optional) You ran the closed-loop policy evaluator (Lab 4 Step 4) here

---

## Tips for Cost Management

1. **Always stop when you walk away.** Set a calendar reminder or use AWS Instance Scheduler.
2. **Use Spot instances for non-critical work.** Modify the CDK stack to use Spot — saves ~70% but can be interrupted.
3. **Right-size the instance.** g6e.4xlarge (1× L40S, 48GB VRAM) is the recommended default for the Isaac Sim AMI and is sufficient for environment development. Change `workstation.instanceType` in `config.json` (or `INSTANCE_TYPE=…`) if you need more — only scale up if you run 4096+ envs visually.
4. **Delete when the project is done.** `cdk destroy PhysicalAi-dev-Workstation` removes everything.

---

**Previous:** [← Lab 1: Train from Demonstrations](lab-1-train-groot.md)
**Next:** [Lab 3: Cosmos World Generation →](lab-3-cosmos-world-generation.md)
