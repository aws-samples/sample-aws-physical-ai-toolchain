#!/bin/bash
# =============================================================================
# Isaac Lab AWS Batch Multi-Node Parallel Training Entrypoint
#
# Translates AWS Batch MNP environment variables to torchrun, mirroring the
# SageMaker entrypoint's distributed logic but reading from Batch's runtime
# env instead of resourceconfig.json.
#
# Environment variables (set by AWS Batch MNP):
#   AWS_BATCH_JOB_NUM_NODES                - Total number of nodes
#   AWS_BATCH_JOB_NODE_INDEX               - This node's index (0-based)
#   AWS_BATCH_JOB_MAIN_NODE_INDEX          - Main node index (typically 0)
#   AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS - Main node IP for rendezvous
#
# Environment variables (set by job definition or nodeOverrides):
#   TASK              - Isaac Lab task name (default: Isaac-Velocity-Flat-Anymal-D-v0)
#   NUM_ENVS          - Parallel environments (default: 4096)
#   MAX_ITERATIONS    - Training iterations (default: 100)
#   PROC_PER_NODE     - Processes (GPUs) per node (default: 4)
#   FRAMEWORK         - RL framework: skrl | rsl_rl | rl_games (default: rsl_rl)
#
# NOTE: Multi-node NCCL convergence is UNVALIDATED on hardware. This script
# wires the topology correctly but has not been run end-to-end on g6 instances.
# =============================================================================

set -e

# --- Read AWS Batch MNP topology ---
NNODES=${AWS_BATCH_JOB_NUM_NODES:-1}
NODE_RANK=${AWS_BATCH_JOB_NODE_INDEX:-0}
MAIN_NODE_INDEX=${AWS_BATCH_JOB_MAIN_NODE_INDEX:-0}
MAIN_IP=${AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS:-localhost}

# --- Count GPUs ---
NPROC=${PROC_PER_NODE:-4}  # job def sets this to match the instance's GPU count

# --- Read hyperparameters from ENV (set by job def or node overrides) ---
TASK="${TASK:-Isaac-Velocity-Flat-Anymal-D-v0}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100}"
FRAMEWORK="${FRAMEWORK:-rsl_rl}"

# --- Print info ---
echo "============================================================"
echo "  Isaac Lab RL Training (AWS Batch MNP)"
echo "  Main node IP:   $MAIN_IP"
echo "  Nodes:          $NNODES"
echo "  Node rank:      $NODE_RANK"
echo "  GPUs per node:  $NPROC"
echo "  Task:           $TASK"
echo "  Num envs:       $NUM_ENVS"
echo "  Max iterations: $MAX_ITERATIONS"
echo "  Framework:      $FRAMEWORK"
echo "============================================================"

# --- Set Isaac Sim environment ---
export ACCEPT_EULA=Y
export OMNI_ENV_PRIVACY_CONSENT=Y

# --- NCCL_SOCKET_IFNAME: prefer autodetect ---
# Some ECS-optimized AMIs use eth0, others use ens5. If you hit NCCL errors like
# "no usable NICs found", uncomment the auto-detect line or set it to the correct
# interface name. Leaving this UNSET lets NCCL autodetect, which usually works.
# To force autodetect at runtime:
# export NCCL_SOCKET_IFNAME=$(ip -o -4 route show to default | awk '{print $5}' | head -1)
# echo "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"

# --- Determine training script based on framework ---
case $FRAMEWORK in
    skrl)
        TRAIN_SCRIPT="scripts/reinforcement_learning/skrl/train.py"
        ;;
    rsl_rl)
        TRAIN_SCRIPT="scripts/reinforcement_learning/rsl_rl/train.py"
        ;;
    rl_games)
        TRAIN_SCRIPT="scripts/reinforcement_learning/rl_games/train.py"
        ;;
    *)
        echo "ERROR: Unknown framework: $FRAMEWORK"
        exit 1
        ;;
esac

# --- EFS symlink for persistent checkpoints ---
# The job definition mounts EFS at /efs. Symlink the framework's output dir to
# /efs/models/<job_id> so checkpoints + tensorboard persist beyond the container
# and are visible from all nodes.
JOB_ID=${AWS_BATCH_JOB_ID:-local}
OUTPUT_DIR="/efs/models/${JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# The RL scripts write logs/rsl_rl/<run> or logs/skrl/<run>. Link that to EFS:
cd /workspace/isaaclab
mkdir -p logs
# Point the framework's log dir to EFS
FRAMEWORK_LOG_DIR="logs/${FRAMEWORK}"
if [ ! -L "$FRAMEWORK_LOG_DIR" ]; then
    mkdir -p "$OUTPUT_DIR/$FRAMEWORK"
    ln -sf "$OUTPUT_DIR/$FRAMEWORK" "$FRAMEWORK_LOG_DIR"
    echo "Linked $FRAMEWORK_LOG_DIR -> $OUTPUT_DIR/$FRAMEWORK"
fi

# --- Launch training via torchrun ---
echo "Launching torchrun (rendezvous endpoint: $MAIN_IP:29500) ..."

/isaac-sim/python.sh -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --rdzv_id=isaaclab-batch-job \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MAIN_IP:29500 \
    $TRAIN_SCRIPT \
    --task=$TASK \
    --num_envs=$NUM_ENVS \
    --max_iterations=$MAX_ITERATIONS \
    --headless

echo "Training complete. Checkpoints saved to $OUTPUT_DIR"
