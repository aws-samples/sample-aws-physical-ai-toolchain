#!/bin/bash
# docker_entrypoint_multi.sh — Isaac Arena connector entrypoint
#
# ARENA_CONNECTOR selects the shipped connector: groot (default).
# Within that evaluator, USE_GROOT_SERVER selects inference or zero-action diagnostics.
# Source the Isaac Sim environment before starting the evaluator.
#
# The connector Dockerfile COPYs this file and sets it as the ENTRYPOINT.
# Editing it requires a connector image rebuild (new immutable tag).

set -e

# Source Isaac Sim environment
if [ -f /isaac-sim/setup_python_env.sh ]; then
    source /isaac-sim/setup_python_env.sh
fi

export CARB_APP_PATH="${CARB_APP_PATH:-/isaac-sim}"
export ISAAC_PATH="${ISAAC_PATH:-/isaac-sim}"
export EXP_PATH="${EXP_PATH:-/isaac-sim/apps}"

# Isaac Sim requires LD_PRELOAD for GPU libs
if [ -f /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 ]; then
    export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0"
fi
ldconfig 2>/dev/null || true

# Accept EULA
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y

echo "[entrypoint] Isaac Sim environment configured"
echo "[entrypoint] USE_GROOT_SERVER=${USE_GROOT_SERVER:-false}"
echo "[entrypoint] ARENA_CONNECTOR=${ARENA_CONNECTOR:-groot}"

# Route to the correct eval path based on ARENA_CONNECTOR
ARENA_CONNECTOR="${ARENA_CONNECTOR:-groot}"
case "${ARENA_CONNECTOR,,}" in
    groot)
        echo "[entrypoint] Running standard eval (GR00T or zero_action)"
        exec python3 /workspace/eval_entry.py
        ;;
    *)
        echo "[entrypoint] ERROR: unsupported ARENA_CONNECTOR='${ARENA_CONNECTOR}'; expected 'groot'." >&2
        exit 2
        ;;
esac
