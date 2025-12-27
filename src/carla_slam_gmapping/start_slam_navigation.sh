#!/bin/bash

# CARLA SLAM + Navigation 启动脚本
# 使用方法：./start_slam_navigation.sh [mode] [town]
#   mode: slam, navigation, slam_navigation
#   town: Town01, Town02, Town03, etc.

set -e

# 颜色输出
RED='\033[0.31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认参数
MODE=${1:-slam}
TOWN=${2:-Town01}
CARLA_PATH=${CARLA_PATH:-"$HOME/carla-0.9.15"}
WS_PATH="$HOME/ros_launch_ws/ros_bridge_ws"

echo -e "${GREEN}=== CARLA SLAM + Navigation System ===${NC}"
echo "Mode: $MODE"
echo "Town: $TOWN"
echo "CARLA Path: $CARLA_PATH"
echo ""

# 检查CARLA路径
if [ ! -d "$CARLA_PATH" ]; then
    echo -e "${RED}Error: CARLA not found at $CARLA_PATH${NC}"
    echo "Set CARLA_PATH environment variable or edit this script"
    exit 1
fi

# 检查CARLA是否已运行
if lsof -Pi :2000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}CARLA server is already running on port 2000${NC}"
else
    echo -e "${YELLOW}Starting CARLA server...${NC}"
    cd "$CARLA_PATH"
    ./CarlaUE4.sh -quality-level=Low -RenderOffScreen &
    CARLA_PID=$!
    echo "CARLA PID: $CARLA_PID"
    echo "Waiting for CARLA to initialize (30 seconds)..."
    sleep 30
fi

# 激活conda环境和设置PYTHONPATH
echo -e "${GREEN}Setting up Python environment...${NC}"
export PYTHONPATH=$PYTHONPATH:$CARLA_PATH/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:$CARLA_PATH/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_PATH/PythonAPI

# 进入ROS工作空间
cd "$WS_PATH"
source devel/setup.bash

# 根据模式启动对应的launch文件
echo -e "${GREEN}Launching ROS nodes (mode: $MODE)...${NC}"

case $MODE in
    slam)
        echo "Starting SLAM mapping system..."
        roslaunch carla_spawn_objects carla_slam_gmapping.launch town:=$TOWN
        ;;
    
    navigation)
        echo "Starting navigation system..."
        echo "Make sure you have a saved map!"
        roslaunch carla_spawn_objects carla_navigation.launch town:=$TOWN
        ;;
    
    slam_navigation)
        echo "Starting simultaneous SLAM + Navigation..."
        roslaunch carla_spawn_objects carla_slam_navigation.launch town:=$TOWN
        ;;
    
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Available modes: slam, navigation, slam_navigation"
        exit 1
        ;;
esac
