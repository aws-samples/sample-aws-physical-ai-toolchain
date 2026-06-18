#!/bin/bash
# Entrypoint for inference container (both Jetson and GPU PC)
#
# Sources ROS 2 environment and launches the inference node.
# Accepts model path and inference rate as arguments.

set -e

# Source ROS 2
source /opt/ros/humble/setup.bash

# Source workspace (if built)
if [ -f /workspace/ros2_ws/install/setup.bash ]; then
    source /workspace/ros2_ws/install/setup.bash
fi

echo "============================================"
echo "  UR3 Pick-and-Place Inference Node"
echo "  Model: ${1:-/model/policy.trt}"
echo "  Rate:  ${2:-200} Hz"
echo "============================================"

# Launch inference node. Prefer `ros2 run` (the workspace builds ur3_inference as
# a proper ament_python package); fall back to running the module directly if the
# workspace wasn't built (e.g. quick local dev).
MODEL_PATH="${1:-/model/policy.trt}"
RATE="${2:-200}"

if ros2 pkg prefix ur3_inference >/dev/null 2>&1; then
    exec ros2 run ur3_inference inference_node --ros-args \
        -p model_path:="$MODEL_PATH" \
        -p inference_rate:="$RATE"
else
    echo "WARN: ur3_inference not found via ros2 — running module directly"
    exec python3 -m ur3_inference.ur3_inference_node --ros-args \
        -p model_path:="$MODEL_PATH" \
        -p inference_rate:="$RATE"
fi
