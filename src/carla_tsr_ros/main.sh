#!/bin/bash
# carla_tsr_ros 启动入口

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 激活 ROS Noetic
if [ -z "$ROS_DISTRO" ]; then
    source /opt/ros/noetic/setup.bash
fi

# 激活 conda 环境（如需）
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate carla38
fi

cd "$SCRIPT_DIR"
exec roslaunch carla_tsr_ros main.launch "$@"
