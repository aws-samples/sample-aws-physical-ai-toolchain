# Physical AI Glossary

A reference for cloud builders entering the robotics/Physical AI space. Organized by category, from foundational concepts to specific tools.

---

## Core Concepts

| Term | What It Is | Cloud Analogy |
|------|-----------|---------------|
| **Physical AI** | AI that interacts with the physical world — robots that can see, reason, and move. Combines perception (cameras), planning (models), and actuation (motors). | Like building an API that talks to hardware instead of databases |
| **Policy** | A trained neural network that takes sensor input (camera image + joint positions) and outputs motor commands (move arm here, close gripper). It's the "brain" of the robot. "Policy" is just the robotics/RL term for "trained model" — the `.pt` file is both. The word "policy" emphasizes *what it does* (maps observations → actions), while "model" emphasizes *what it is* (architecture + weights). When a system has multiple models (e.g., a VLM for planning + a diffusion model for control), "the policy" usually refers specifically to the one outputting motor commands. | Like a Lambda function: input → computation → output. But the output moves physical things. |
| **Embodiment** | The physical robot body. Different robots have different joint counts, sensor setups, and capabilities. A policy trained for one embodiment may not work on another. | Like platform-specific code — your ARM binary won't run on x86 without recompilation. |
| **Sim-to-Real Transfer** | Training a policy in simulation, then deploying it on a real robot. The gap between sim physics and real-world physics is the main challenge. | Like testing in staging vs. production — sim is staging, but the "hardware" is different. |
| **Domain Randomization** | During training in sim, randomly vary textures, lighting, object sizes, physics parameters. This forces the policy to generalize, improving sim-to-real transfer. | Like chaos engineering — inject randomness during training so the model is robust to real-world variation. |

---

## Models & Training Approaches

| Term | What It Is | When to Use |
|------|-----------|-------------|
| **VLA (Vision-Language-Action) Model** | One AI model that sees images + understands language + outputs robot motor commands. GR00T and π0 are VLAs. | When you want one model that understands "pick up the red cube" from camera input. |
| **VLM (Vision-Language Model)** | Sees images + reasons in text, but does NOT output motor commands (e.g., Claude Sonnet on Bedrock). Can be used to generate reward functions or plan high-level actions. | For high-level reasoning about scenes, not direct robot control. |
| **GR00T** | NVIDIA's VLA foundation model (Generalist Robot 00 Technology). Pre-trained on diverse robot data. You fine-tune it on your specific robot + task. Current version: N1.7-3B (3 billion parameters). Supports robot arms and humanoids. | Default choice for this toolchain. Fine-tune with 50-100 teleoperated demonstrations. |
| **π0 (Pi Zero)** | Physical Intelligence's open-source VLA foundation model. Similar to GR00T — pre-trained, fine-tunable. More beginner-friendly, supports GR00T-style tasks. | If you want a non-NVIDIA option. Same workflow, different base model. |
| **Foundation Model** | A large pre-trained model you fine-tune for your specific task. Like an LLM but for robot vision and motor commands. GR00T and π0 are foundation models. | Always — you fine-tune, never train from scratch. |
| **Reinforcement Learning (RL)** | Robot learns by trial and error in simulation. Given a reward function ("you get +1 when you pick up the object"), it discovers how to do the task through millions of attempts. | When you don't have demonstration data, or want the robot to discover novel strategies. |
| **Imitation Learning** | Robot learns by watching demonstrations (teleop recordings or human videos). Simpler than RL but limited to behaviors shown in the data. | When you can easily demonstrate the task. Faster to get working than RL. |
| **PPO (Proximal Policy Optimization)** | The most common RL algorithm for robotics. Stable, well-understood, works across many tasks. | Default RL algorithm in Isaac Lab. Used in your `ppo_pick_place.yaml` config. |
| **Fine-tuning** | Taking a pre-trained model (GR00T) and training just the last few layers on your specific data. Much cheaper than training from scratch. | Always — you never train GR00T from scratch. You fine-tune the projector + diffusion head. |
| **ACT (Action Chunking Transformer)** | Policy architecture that predicts sequences of future actions (a "chunk"), not just one at a time. Smoother motion, fewer compounding errors. | Used in the Isaac Sim imitation learning path. |
| **Diffusion Policy** | Alternative policy architecture that generates smooth action trajectories using diffusion math (like DALL-E but for robot movements instead of images). | GR00T uses this internally. You don't choose it — it's built into the model. |
| **Action Chunking** | Instead of predicting one action at a time, the model predicts the next N actions (e.g., 16 steps). Smoother motion, less jitter. | GR00T outputs 16-step action chunks. The inference node executes them sequentially. |

---

## Data & Formats

| Term | What It Is | File Extension |
|------|-----------|---------------|
| **LeRobot** | HuggingFace's standard dataset format for robot learning. Parquet files for states/actions + MP4 videos for camera frames. Used by GR00T. | `.parquet`, `.mp4`, `info.json` |
| **MCAP** | Container file format for timestamped robot data (cameras, joints, sensors). The default bag format in ROS 2. Think "MP4 for robot data." | `.mcap` |
| **Zarr** | Chunked array storage format. Used for storing raw teleoperation episodes locally before conversion to LeRobot format. | `.zarr` (directory) |
| **URDF** | Universal Robot Description Format. An XML file that describes a robot's geometry (links, joints, dimensions, limits). Every simulator needs this. | `.urdf` |
| **USD / OpenUSD** | Universal Scene Description. Originally created by Pixar, now an open standard (OpenUSD) governed by the Alliance for OpenUSD (AOUSD — Apple, NVIDIA, Pixar, Adobe, Autodesk). It's a file format *and* runtime for composing, describing, and collaborating on 3D scenes — geometry, materials, physics properties, lighting, animations, and spatial relationships. Think of it as "the HTML of 3D" — a universal interchange format that different tools (Isaac Sim, Blender, CAD software, game engines) can all read and write. Isaac Sim uses USD natively for all scene content. | `.usd`, `.usda` (text), `.usdc` (binary) |
| **Teleoperation (Teleop)** | A human remotely controlling a robot to demonstrate tasks. The recordings become training data. | — |
| **Episode** | One complete task execution from start to finish (e.g., one pick-and-place cycle). A dataset contains many episodes. | — |

---

## Data Management & Augmentation

| Term | What It Is | Use in This Toolchain |
|------|-----------|----------------------|
| **NeMo Curator** | NVIDIA's GPU-accelerated data curation platform. Handles text, image, and video at scale. For video: clip splitting (fixed stride or scene-change detection), encoding, embedding generation, deduplication, quality/motion filtering. Runs on Ray, scales from laptop to multi-node GPU cluster. | Preprocessing raw robot video data — deduplicate, filter low-quality clips, generate embeddings for similarity search. |
| **MimicGen / GR00T-Mimic** | Synthetic trajectory augmentation system. Takes a small set of human teleop demonstrations (~200) and automatically generates 50K+ synthetic trajectories by adapting them to new object positions, scene configurations, and robot arms. Available as a NIM microservice and as the "Isaac GR00T-Mimic Blueprint." | Multiplying limited real demonstration data — turns 200 demos into 50K+ training trajectories without additional human effort. |
| **Robocasa** | NIM microservice that procedurally generates robot tasks and simulation-ready environments in OpenUSD. Produces diverse household/kitchen scenes with randomized objects, layouts, and tasks. | Environment diversity — generates varied USD scenes so manipulation policies generalize beyond one setting. |
| **Physical AI Data Factory Blueprint** | NVIDIA's end-to-end reference architecture (announced GTC 2025) that unifies data generation, augmentation, and evaluation into one orchestrated pipeline. Combines NeMo Curator + Cosmos + MimicGen + Isaac Lab. | The "big picture" pipeline — automates the full data lifecycle from raw video to curated training set. |
| **NeMo Agent Toolkit (Physical AI Skills)** | Multi-agent workflows using LLMs to automate dataset creation. Agents use Omniverse, Cosmos, and NIM to generate scenes, place objects, run simulations, and produce annotated datasets — no manual scripting. | Automating synthetic data generation — describe what you want in natural language, agents produce the training data. |
| **Foxglove** | Commercial platform for robotics data visualization, debugging, and management. Supports live and recorded data (MCAP, ROS bags, Protobuf, JSON) with 20+ visualization panels (3D, images, plots, logs). Includes a cloud Data Platform for centralized storage, search, and team collaboration on recordings. Originally open-source (Foxglove Studio), now a commercial product. | Primary use: visualize and debug robot sensor data (camera feeds, lidar, joint states). Secondary: centralized log management and search across fleet recordings. |
| **Rerun** | Open-source SDK + viewer for logging, storing, querying, and visualizing multimodal time-series data (images, point clouds, transforms, joint states, video). Supports C++, Python, and Rust. Ingests MCAP, LeRobot, and its own `.rrd` format. Lightweight and embeddable — designed for dev-time debugging rather than fleet-scale management. | Debug-time visualization during training/eval — log what the policy sees and does, replay it visually. |
| **Roboto AI** | Cloud analytics platform for robotics data at scale. Ingests any file format (MCAP, ROS bags, etc.), provides AI-powered search across logs, anomaly detection, automated QA actions, and event highlighting. Think "Datadog for robot data." | Fleet-scale log analytics — find failure cases, surface edge cases, root-cause issues across thousands of recordings. |
| **ReductStore** | Open-source time-series object store designed for high-frequency binary data (images, sensor streams, MCAP files). Optimized for edge-to-cloud replication with limited bandwidth — stores data on-robot, syncs when connected. | On-device data capture and selective upload — store everything locally, replicate interesting segments to cloud. |
| **Robo-DM** | Research data format (Berkeley) for robot datasets. Uses EBML (Extensible Binary Meta Language) for self-contained, compressed storage of robot trajectories. Reduces dataset size, transfer costs, and training data load times vs. raw formats. | Alternative dataset format — more compact than raw MCAP/Zarr for large-scale trajectory datasets. |

---

## Simulation & Physics

| Term | What It Is | Use in This Toolchain |
|------|-----------|----------------------|
| **Isaac Sim** | NVIDIA's full simulation *platform* built on Omniverse. Provides physics (PhysX), photorealistic rendering (RTX), sensor simulation (cameras, lidar, IMU), and digital-twin capabilities. Think of it as the "engine" — it does rendering, physics, and scene management but is agnostic to what you run on top. | Scene generation, evaluation video rendering, sensor-sim for synthetic data. Isaac Lab runs *on top of* Isaac Sim. |
| **Omniverse** | NVIDIA's platform for 3D simulation and collaboration. Isaac Sim runs on top of it. Think of it as the "OS" that hosts Isaac Sim. | You don't interact with it directly — Isaac Sim abstracts it. |
| **Isaac Lab** | A lightweight *RL/robot-learning framework* that runs on top of Isaac Sim (formerly called "Orbit"). It provides Gym-style vectorized environments, reward/observation APIs, pre-built tasks (locomotion, manipulation), and integration with RL libraries (RSL-RL, rl_games, Stable Baselines). Isaac Lab is to Isaac Sim what PyTorch Lightning is to PyTorch — a higher-level training harness, not a simulator itself. | Where the actual RL training loop runs (`training/envs/pick_and_place_ur3.py`). You write env configs here; Isaac Sim handles the physics underneath. |
| **Cosmos** | NVIDIA's *World Foundation Model platform* for Physical AI. Not a single model — it's an end-to-end platform with multiple components (see below). Generates physics-aware synthetic video/scenes for training robots and AVs. | Synthetic data generation — diverse training environments so the policy doesn't overfit to one scene. Also used for future-state prediction and domain transfer. |
| **Cosmos Tokenizer** | A component of the Cosmos platform. Encodes raw video frames into compact latent tokens that the WFMs can process, and decodes tokens back into video. Achieves high spatial and temporal compression while preserving visual fidelity. Operates in continuous (for diffusion models) or discrete (for autoregressive models) modes. | Preprocessing layer — compresses video data before the WFMs operate on it. |
| **Cosmos Predict** | The generalist World Foundation Model within Cosmos. Given text, image, or video input, it generates future video frames that obey physics. Comes in autoregressive and diffusion variants at multiple sizes (Nano, Super). Trained on 20M+ hours of physical world video. | Core generation engine — produce synthetic training videos of robots/environments. |
| **Cosmos Transfer** | A controllable generation model that transforms structured inputs (depth maps, segmentation masks, edge maps) into photorealistic video. Enables precise control over generated scene content while maintaining realism. | Domain transfer — take a sim render and produce a photorealistic version, or stylize scenes for domain randomization. |
| **Cosmos Reason** | Vision-language model component that can perceive video, reason about physical interactions (object dynamics, spatial relationships), and respond to queries about the world state. | Understanding layer — evaluate whether generated scenes are physically plausible, or describe what's happening. |
| **Cosmos Guardrail** | Safety filtering component that screens both inputs and outputs for inappropriate or unsafe content during world generation. | Content safety — ensures generated training data doesn't contain problematic content. |
| **NeMo Curator (for Cosmos)** | NVIDIA's GPU-accelerated data curation pipeline for video. Handles deduplication, quality filtering, captioning, and metadata tagging of large video datasets before they're used to train or fine-tune Cosmos WFMs. | Data prep — curate the raw video corpus before training/fine-tuning. |
| **MuJoCo** | DeepMind's physics simulator. Older than Isaac Sim, less visual fidelity, but fast and well-trusted in research. | Not used in this toolchain (Isaac Lab is our sim). Mentioned because LeRobot references it. |
| **Gazebo** | Lightweight open-source robot simulator. ROS-native. Fast iteration, not photorealistic. Good for quick prototyping. | Used in the hackathon repo for quick prototyping. Not in the primary paths here. |
| **PhysX** | NVIDIA's physics engine (collisions, friction, gravity). The "engine under the hood" of Isaac Sim. | You don't interact with it directly — Isaac Lab abstracts it. |
| **DCV (NICE Desktop Cloud Visualization)** | AWS remote desktop protocol for GPU-accelerated graphics. Lets you see Isaac Sim's UI from your laptop by streaming pixels from a GPU EC2 instance. | Optional: for debugging environments visually. Not needed for headless training. |
| **Parallel Environments** | Running 4096 copies of the same robot simultaneously on one GPU. Each copy tries different actions. Massively speeds up RL training. | That's why training takes 4-8 hours instead of months. |
| **Headless** | Running the simulator without rendering graphics. All physics still work, but no screen output. Faster because GPU isn't spending time on pixels. | Default during training. Rendering only needed for eval videos. |

---

## Edge & Deployment

| Term | What It Is | AWS Equivalent |
|------|-----------|---------------|
| **TensorRT** | NVIDIA's model compiler. Takes a PyTorch model, optimizes it for specific GPU hardware, outputs a binary engine that runs 10-100x faster. | Like compiling Java to native code — same logic, way faster execution. |
| **ONNX** | Open Neural Network Exchange. An intermediate format between PyTorch and TensorRT. Export PyTorch → ONNX → compile to TensorRT. | Like LLVM IR — a portable intermediate before the final hardware-specific binary. |
| **Jetson** | NVIDIA's edge GPU boards (Orin, Xavier, Thor). Small computers with GPUs designed for robots, drones, and edge AI. Runs TensorRT at low power. | Like running inference on a tiny Lambda with a GPU attached, physically inside the robot. |
| **ROS 2** (Robot Operating System) | Industry-standard middleware for robot communication. Pub/sub messaging between sensors, planners, and actuators. Not actually an OS — more like Kubernetes for robots. | Like SNS/SQS for robot components. Nodes publish sensor data, subscribe to commands. |
| **MoveIt** | Motion planning library for robot arms. Give it a target position, it calculates collision-free joint movements using inverse kinematics. | Like a route planner but for robot arm joints instead of roads. |
| **Inverse Kinematics (IK)** | Math that calculates what joint angles are needed to put the hand at a desired position. MoveIt handles this. | Like DNS resolution — you give it a name (position), it figures out the address (joint angles). |
| **Greengrass** | AWS IoT Greengrass. Deploys and manages software on edge devices (Jetsons). OTA updates, fleet management, health monitoring. | This is your deployment mechanism — pushes new models from S3 to robot fleet. |
| **Inference Rate (Hz)** | How many times per second the policy runs. 200 Hz = 200 decisions/second. Real-time control needs >100 Hz. | Like request latency — but instead of API response time, it's "how fast can the robot react." |
| **UR3 / UR5** | Universal Robots industrial arms (6 joints). UR3: 3kg payload, tabletop. UR5: 5kg, larger reach. Same ROS 2 driver. Most common arm in manufacturing and research. | — |

---

## Orchestration

| Term | What It Is | AWS Equivalent |
|------|-----------|---------------|
| **OSMO** | NVIDIA's workflow orchestration for Physical AI. Manages multi-stage pipelines (generate scenes → train → evaluate → export). Runs on Kubernetes. | Like Step Functions or SageMaker Pipelines, but designed specifically for GPU-heavy robotics workflows. |
| **KAI Scheduler** | NVIDIA's GPU-aware Kubernetes scheduler. Understands GPU topology, handles preemption, and enables multi-tenant GPU sharing. | Like Karpenter but GPU-specific — knows which pods need which GPU types. |
| **SageMaker Training Jobs** | AWS fully managed training. You give it a container + data + instance type → it provisions hardware, trains, uploads results, terminates. No EKS needed. | This is Path A's approach. Simpler than OSMO for single-stage training. |
| **SageMaker Pipelines** | DAG orchestration for ML workflows on SageMaker. Chain multiple training/processing steps with data dependencies. | Could replace OSMO for simpler multi-stage pipelines. |

---

## Acronyms

| Acronym | Full Name |
|---------|-----------|
| VLA | Vision-Language-Action (model) |
| RL | Reinforcement Learning |
| PPO | Proximal Policy Optimization |
| DOF | Degrees of Freedom (number of joints a robot has) |
| EE | End Effector (the gripper or tool at the end of the arm) |
| IK | Inverse Kinematics (calculating joint angles to reach a position) |
| FK | Forward Kinematics (calculating position from joint angles) |
| OTA | Over-The-Air (update) |
| WFM | World Foundation Model (Cosmos) |
| MoT | Mixture-of-Transformers (Cosmos 3 architecture) |
| FP16 | Half-precision floating point (faster inference, slight accuracy trade) |
| IRSA | IAM Roles for Service Accounts (EKS ↔ AWS IAM bridge) |
| AOUSD | Alliance for OpenUSD (Apple, NVIDIA, Pixar, Adobe, Autodesk) |
| EBML | Extensible Binary Meta Language (used by Robo-DM format) |

---

## Tools in This Repo

| Tool | Layer | What It Does Here |
|------|-------|-------------------|
| AWS CDK | Infrastructure | Deploys all AWS resources (S3, ECR, EKS, IAM) |
| SageMaker | Training (Path A) | Runs GR00T fine-tuning on managed GPU instances |
| EKS | Training (Path B) | Hosts Isaac Lab + OSMO for sim-based RL |
| S3 | Storage | Datasets, checkpoints, trained models |
| ECR | Containers | Stores Docker images for training + inference |
| CodeBuild | CI | Builds container images from Dockerfiles |
| IoT Core | Edge | Device registry and MQTT messaging for robots |
| Greengrass | Edge | Deploys inference containers to Jetson fleet |
| CloudWatch / MLflow | Monitoring | Training metrics, loss curves, logs |

---

## Further Reading

- [NVIDIA Isaac Lab docs](https://isaac-sim.github.io/IsaacLab/)
- [LeRobot documentation](https://huggingface.co/docs/lerobot)
- [GR00T fine-tuning guide](https://nvidia-isaac-gr00t.mintlify.app/guides/finetuning)
- [ROS 2 concepts](https://docs.ros.org/en/humble/Concepts.html)
- [AWS IoT Greengrass developer guide](https://docs.aws.amazon.com/greengrass/v2/developerguide/)
- [Universal Robots UR3 specs](https://www.universal-robots.com/products/ur3-robot/)
