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

# Launch inference node
exec python3 /workspace/ros2_ws/src/ur3_inference/ur3_inference_node.py \
    --ros-args \
    -p model_path:="${1:-/model/policy.trt}" \
    -p inference_rate:="${2:-200}" \
    "$@"
