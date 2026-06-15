#!/bin/bash
# =============================================================================
# Isaac Lab SageMaker Training Entrypoint
#
# Based on the official AWS pattern from:
# https://github.com/awslabs/awsome-distributed-ai/tree/main/3.test_cases/pytorch/nvidia-isaac-lab
#
# Parses SageMaker's resource config for multi-node topology, then launches
# Isaac Lab RL training via torchrun.
#
# Environment variables (set by SageMaker or hyperparameters):
#   TASK              - Isaac Lab task name (default: Isaac-Velocity-Flat-Anymal-D-v0)
#   NUM_ENVS          - Parallel environments (default: 4096)
#   MAX_ITERATIONS    - Training iterations (default: 100)
#   FRAMEWORK         - RL framework: skrl | rsl_rl | rl_games (default: rsl_rl)
# =============================================================================

set -e

# --- Read SageMaker resource config for multi-node setup ---
RESOURCE_CONFIG="/opt/ml/input/config/resourceconfig.json"

if [ -f "$RESOURCE_CONFIG" ]; then
    CURRENT_HOST=$(python3 -c "import json; c=json.load(open('$RESOURCE_CONFIG')); print(c['current_host'])")
    ALL_HOSTS=$(python3 -c "import json; c=json.load(open('$RESOURCE_CONFIG')); print(','.join(c['hosts']))")
    MASTER_HOST=$(python3 -c "import json; c=json.load(open('$RESOURCE_CONFIG')); print(c['hosts'][0])")
    NNODES=$(python3 -c "import json; c=json.load(open('$RESOURCE_CONFIG')); print(len(c['hosts']))")
    NODE_RANK=$(python3 -c "import json; c=json.load(open('$RESOURCE_CONFIG')); print(c['hosts'].index(c['current_host']))")
else
    CURRENT_HOST="localhost"
    MASTER_HOST="localhost"
    NNODES=1
    NODE_RANK=0
fi

# --- Count GPUs ---
NPROC=$(nvidia-smi -L | wc -l)

# --- Read hyperparameters ---
HYPERPARAMS="/opt/ml/input/config/hyperparameters.json"
if [ -f "$HYPERPARAMS" ]; then
    TASK=$(python3 -c "import json; print(json.load(open('$HYPERPARAMS')).get('task', 'Isaac-Velocity-Flat-Anymal-D-v0'))")
    NUM_ENVS=$(python3 -c "import json; print(json.load(open('$HYPERPARAMS')).get('num_envs', '4096'))")
    MAX_ITERATIONS=$(python3 -c "import json; print(json.load(open('$HYPERPARAMS')).get('max_iterations', '100'))")
    FRAMEWORK=$(python3 -c "import json; print(json.load(open('$HYPERPARAMS')).get('framework', 'rsl_rl'))")
else
    TASK="${TASK:-Isaac-Velocity-Flat-Anymal-D-v0}"
    NUM_ENVS="${NUM_ENVS:-4096}"
    MAX_ITERATIONS="${MAX_ITERATIONS:-100}"
    FRAMEWORK="${FRAMEWORK:-rsl_rl}"
fi

# --- Print info ---
echo "============================================================"
echo "  Isaac Lab RL Training (SageMaker)"
echo "  Host:           $CURRENT_HOST"
echo "  Master:         $MASTER_HOST"
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

# --- Launch training via torchrun ---
cd /workspace/isaaclab

/isaac-sim/python.sh -m torch.distributed.run \
    --nproc_per_node=$NPROC \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --rdzv_id=isaaclab-sm-job \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_HOST:29500 \
    $TRAIN_SCRIPT \
    --task=$TASK \
    --num_envs=$NUM_ENVS \
    --max_iterations=$MAX_ITERATIONS \
    --headless

# --- Copy artifacts to SageMaker output ---
echo "Copying training artifacts to /opt/ml/model/..."
if [ -d "logs" ]; then
    cp -r logs /opt/ml/model/
fi

# Save metadata
python3 -c "
import json
meta = {
    'task': '$TASK',
    'num_envs': int('$NUM_ENVS'),
    'max_iterations': int('$MAX_ITERATIONS'),
    'framework': '$FRAMEWORK',
    'nodes': int('$NNODES'),
    'gpus_per_node': int('$NPROC'),
}
json.dump(meta, open('/opt/ml/model/training_metadata.json', 'w'), indent=2)
"

echo "Done. Artifacts saved to /opt/ml/model/"
