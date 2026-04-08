#!/bin/bash
# CARLA SLAM GMMapping 自动化安装脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "  CARLA SLAM GMMapping 安装脚本"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查是否在ROS工作空间中
check_ros_workspace() {
    if [ ! -f "../../CMakeLists.txt" ]; then
        echo -e "${RED}错误: 请在ROS工作空间的src目录中运行此脚本${NC}"
        echo "例如: cd ~/catkin_ws/src/carla_slam_gmapping && ./install.sh"
        exit 1
    fi
}

# 检查ROS是否安装
check_ros_installation() {
    echo -e "${YELLOW}[1/6] 检查ROS安装...${NC}"
    if [ -z "$ROS_DISTRO" ]; then
        echo -e "${RED}错误: 未检测到ROS环境变量${NC}"
        echo "请先source ROS环境:"
        echo "  source /opt/ros/noetic/setup.bash"
        exit 1
    fi
    echo -e "${GREEN}✓ ROS $ROS_DISTRO 已安装${NC}"
}

# 安装ROS依赖
install_ros_dependencies() {
    echo -e "${YELLOW}[2/6] 安装ROS包依赖...${NC}"
    
    # 返回到工作空间根目录
    cd ../..
    
    # 使用rosdep安装依赖
    if command -v rosdep &> /dev/null; then
        echo "使用rosdep安装依赖..."
        rosdep update
        rosdep install --from-paths src --ignore-src -r -y
    else
        echo -e "${YELLOW}警告: rosdep未安装，手动安装必需包...${NC}"
        sudo apt-get update
        sudo apt-get install -y \
            ros-$ROS_DISTRO-gmapping \
            ros-$ROS_DISTRO-map-server \
            ros-$ROS_DISTRO-move-base \
            ros-$ROS_DISTRO-dwa-local-planner \
            ros-$ROS_DISTRO-global-planner \
            ros-$ROS_DISTRO-carla-ros-bridge \
            ros-$ROS_DISTRO-carla-msgs \
            ros-$ROS_DISTRO-rviz
    fi
    
    echo -e "${GREEN}✓ ROS依赖安装完成${NC}"
    
    # 返回到包目录
    cd src/carla_slam_gmapping
}

# 安装Python依赖
install_python_dependencies() {
    echo -e "${YELLOW}[3/6] 安装Python依赖...${NC}"
    
    if [ ! -f "requirements.txt" ]; then
        echo -e "${RED}错误: requirements.txt 文件不存在${NC}"
        exit 1
    fi
    
    # 检查pip
    if ! command -v pip3 &> /dev/null; then
        echo "安装pip3..."
        sudo apt-get install -y python3-pip
    fi
    
    # 安装Python包
    pip3 install -r requirements.txt
    
    echo -e "${GREEN}✓ Python依赖安装完成${NC}"
}

# 安装CARLA Python API
install_carla_api() {
    echo -e "${YELLOW}[4/6] 检查CARLA Python API...${NC}"
    
    # 检查CARLA是否已安装
    if python3 -c "import carla" 2>/dev/null; then
        echo -e "${GREEN}✓ CARLA Python API 已安装${NC}"
    else
        echo -e "${YELLOW}警告: CARLA Python API 未安装${NC}"
        echo "请手动安装CARLA Python API:"
        echo "1. 下载CARLA: https://github.com/carla-simulator/carla/releases"
        echo "2. 安装egg文件:"
        echo "   pip3 install /path/to/carla/PythonAPI/carla/dist/carla-0.9.15-py3.x-linux-x86_64.egg"
        echo ""
        read -p "是否跳过此步骤继续? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# 编译工作空间
build_workspace() {
    echo -e "${YELLOW}[5/6] 编译ROS工作空间...${NC}"
    
    # 返回到工作空间根目录
    cd ../..
    
    # 使用catkin_make编译
    if command -v catkin_make &> /dev/null; then
        catkin_make
    elif command -v catkin &> /dev/null; then
        catkin build carla_slam_gmapping
    else
        echo -e "${RED}错误: 未找到catkin编译工具${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ 编译完成${NC}"
}

# 配置环境
setup_environment() {
    echo -e "${YELLOW}[6/6] 配置环境...${NC}"
    
    # 创建必要的目录
    cd src/carla_slam_gmapping
    mkdir -p maps logs
    
    echo -e "${GREEN}✓ 环境配置完成${NC}"
}

# 显示安装完成信息
show_completion_message() {
    echo ""
    echo "=========================================="
    echo -e "${GREEN}  安装完成！${NC}"
    echo "=========================================="
    echo ""
    echo "下一步操作:"
    echo ""
    echo "1. 更新环境变量:"
    echo "   source ~/catkin_ws/devel/setup.bash"
    echo ""
    echo "2. 启动CARLA模拟器 (另开终端):"
    echo "   cd /path/to/carla"
    echo "   ./CarlaUE4.sh"
    echo ""
    echo "3. 启动ROS Bridge (另开终端):"
    echo "   roslaunch carla_ros_bridge carla_ros_bridge.launch"
    echo ""
    echo "4. 启动SLAM导航系统:"
    echo "   roslaunch carla_slam_gmapping carla_slam_navigation.launch"
    echo ""
    echo "5. 或使用快速启动脚本:"
    echo "   ./start_slam_navigation.sh"
    echo ""
    echo "=========================================="
}

# 主函数
main() {
    check_ros_workspace
    check_ros_installation
    install_ros_dependencies
    install_python_dependencies
    install_carla_api
    build_workspace
    setup_environment
    show_completion_message
}

# 运行主函数
main
