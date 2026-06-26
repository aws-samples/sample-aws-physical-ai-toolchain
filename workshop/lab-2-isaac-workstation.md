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
| 5 | Launch Isaac Sim (visual) | `isaac-sim.sh` on the DCV desktop | 3D viewport opens |
| 6 | Test the training container (headless) | `docker run --gpus all …isaac-lab:latest train` | training loop logs steps/s |
| 7 | (Optional) Closed-loop policy eval | `pai eval serve` + `pai eval --closed-loop` (two terminals) — see *Closed-Loop Policy Evaluation* | prints `success_rate` JSON (**unvalidated on GPU**) |
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

> **Note:** The install path (`/opt/IsaacSim`) is defined by the AMI and verified on the current Isaac Sim Marketplace release. If a future AMI moves it, find the launcher with `ls /opt/IsaacSim/isaac-sim.sh ~/IsaacSim/isaac-sim.sh`. The **Docker container path** (see "Testing Training Containers Locally" below) remains the verified, reliable route for *headless training* and achieves ~56k steps/s (measured on A10G; the L40S on g6e is faster).

You'll see the full Isaac Sim visual editor — 3D viewport, content browser with robots and environments, scene tree.

---

## Step 4: Develop Your RL Environment

This is where you iterate. Isaac Sim is pre-installed via the AMI (Step 3); for *headless training* the **verified, working route** is the Docker container (see "Testing Training Containers Locally" below).

**If the host GUI install worked:**

```bash
# The toolchain code is ALREADY on the workstation. The deploy bundles your local
# working tree as an S3 asset and the bootstrap unzips it to:
#   /home/ubuntu/aws-physical-ai-toolchain
# (The public GitHub repo isn't released yet, so there's no git clone — once it's
# public you can pass --context repoUrl=<url> to clone instead.)
cd ~/aws-physical-ai-toolchain

# Run the UR3 pick-and-place environment visually
./isaaclab.sh -p training/envs/pick_and_place_ur3.py --num_envs=8

# Watch the robot attempt the task
# Tweak rewards, observation space, action space
# When it looks right → launch headless on SageMaker (Lab 4)
```

**If the host GUI install didn't complete (or you want the reliable path):**

Use the local Docker container (cross-reference "Testing Training Containers Locally" below). The UR3 environment is now wired into the container's `train` entrypoint, so running the container locally gives you headless UR3 training at ~56k steps/s on A10G — same code that will run on SageMaker in Lab 4.

**What to look for (if visual rendering works):**
- Robot reaching toward the object (approach reward working)
- Gripper closing at the right time (grasp reward working)
- Object being lifted cleanly (success reward working)
- No physics glitches (objects clipping through surfaces)
- Domain randomization looking reasonable (not too wild)

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

## Testing Training Containers Locally

> **Isaac Sim vs. `isaac-lab` image — don't confuse them.** Two different things:
> - **Isaac Sim** (the GUI simulator, Step 3) is **pre-installed in the Marketplace AMI** at `/opt/IsaacSim`. Nothing to pull.
> - **`physical-ai/isaac-lab:latest`** (below) is *this project's own* headless training container — the one SageMaker runs in Lab 4. It's built by **CodeBuild into your ECR when you deploy the Foundation stack**, and is unrelated to the AMI's Isaac Sim.
>
> So this step only works **after** the Foundation stack has been deployed in *this* account/region and CodeBuild has finished building the image. If you haven't deployed Foundation yet, the `docker pull` below returns `not found` — that's expected; finish Lab 1 (or deploy Foundation) first. (The workstation bootstrap pre-pulls this image best-effort and logs a non-fatal "image not in ECR yet" if it's missing.)

The workstation has Docker + NVIDIA Container Toolkit, so you can pull the
CodeBuild-built image from *your* ECR and run it exactly as SageMaker would —
without waiting for SageMaker provisioning. The registry is derived from your own
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

This is the home of **Lab 4's closed-loop evaluator**. Open-loop eval (policy runs
in-process with the env) is fine for a quick number, but it doesn't exercise the
*serving* path. The closed-loop evaluator splits the policy and the simulator into
two processes that talk over ZMQ — a **policy server** answers observation→action
requests, and a **sim client** drives Isaac Lab step-by-step — which mirrors how the
policy is actually served on the robot (Lab 5). It needs the L40S GPU on this
workstation, so it lives here rather than on your laptop.

> ⚠️ **UNVALIDATED on hardware.** The whole Isaac Lab GPU path (env instantiation +
> rendering) has not been run end-to-end on this workstation yet. The code is wired
> and CI-tested with mocks; treat the numbers below as the *expected* shape, not a
> measured result.

**Prerequisites on the workstation:**
- The toolchain code is already at `/home/ubuntu/aws-physical-ai-toolchain` (the
  deploy bundles it). `pip install -r training/requirements.txt` adds `pyzmq`.
- A **TorchScript** checkpoint. Raw RL checkpoints (rsl_rl state dicts) will NOT
  load — scriptify first with `export.py` (see Lab 4 Step 4a). The server fails
  loudly if you hand it a raw `.pt`.

**Terminal 1 — policy server** (loads the model, answers action requests):

```bash
cd /home/ubuntu/aws-physical-ai-toolchain
pai eval serve --checkpoint ./model_scripted/model_scripted.pt --device cuda
# Binds tcp://127.0.0.1:5555 (localhost only — ZMQ has no auth; don't bind 0.0.0.0)
```

**Terminal 2 — sim client** (drives Isaac Lab, records success/failure):

```bash
cd /home/ubuntu/aws-physical-ai-toolchain
pai eval --closed-loop \
  --env PickAndPlaceUR3-v0 \
  --endpoint tcp://127.0.0.1:5555 \
  --eval-rounds 100 \
  --output-dir ./eval_results
```

The client writes the same JSON metrics schema as the open-loop evaluator
(`success_rate_pct`, `num_episodes`, `avg_reward`, `avg_cycle_time_sec`,
`failure_modes{timeout,drop,collision}`). Full walkthrough, including scriptifying
the checkpoint, is in **[Lab 4 → Step 4](lab-4-rl-refinement.md#step-4-evaluate-the-trained-policy-closed-loop)**.

<details>
<summary>Under the hood (raw commands)</summary>

```bash
# Terminal 1
python training/scripts/eval_policy_server.py \
  --checkpoint ./model_scripted/model_scripted.pt \
  --device cuda

# Terminal 2
python training/scripts/eval_sim_client.py \
  --task PickAndPlaceUR3-v0 \
  --endpoint tcp://127.0.0.1:5555 \
  --eval-rounds 100 \
  --output-dir ./eval_results
```

</details>

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
