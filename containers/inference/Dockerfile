# Inference Container for Edge Deployment
#
# Lightweight container with:
# - TensorRT runtime (runs compiled .trt policy)
# - ROS 2 Humble (communicates with UR3 via ros2_control)
# - Minimal dependencies (fast startup, small image)
#
# Targets:
# - Jetson Orin (aarch64) — via JetPack base image
# - GPU PC (x86_64) — via NVIDIA TensorRT base image
#
# Build (x86_64):
#   docker build -t inference:latest --target x86 .
#
# Build (Jetson/aarch64):
#   docker build -t inference:latest --target jetson .

# ============================================================
# Stage: x86_64 (GPU PC)
# ============================================================
FROM nvcr.io/nvidia/tensorrt:24.05-py3 AS x86

# Install ROS 2 Humble
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    && add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update && apt-get install -y \
    ros-humble-ros-base \
    ros-humble-ur-robot-driver \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy>=1.24 \
    pyyaml>=6.0

# Copy inference node
COPY edge/ros2-workspace/ /workspace/ros2_ws/src/
COPY edge/scripts/ /workspace/scripts/

WORKDIR /workspace

# Build ROS 2 workspace
RUN . /opt/ros/humble/setup.sh && \
    cd ros2_ws && \
    colcon build --symlink-install

# Entrypoint: source ROS 2 and run inference node
COPY edge/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# ============================================================
# Stage: Jetson (aarch64)
# ============================================================
FROM nvcr.io/nvidia/l4t-tensorrt:r36.4.0 AS jetson

# ROS 2 Humble is available on JetPack 6.x
RUN apt-get update && apt-get install -y \
    ros-humble-ros-base \
    ros-humble-ur-robot-driver \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy>=1.24 \
    pyyaml>=6.0

COPY edge/ros2-workspace/ /workspace/ros2_ws/src/
COPY edge/scripts/ /workspace/scripts/

WORKDIR /workspace

RUN . /opt/ros/humble/setup.sh && \
    cd ros2_ws && \
    colcon build --symlink-install

COPY edge/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
