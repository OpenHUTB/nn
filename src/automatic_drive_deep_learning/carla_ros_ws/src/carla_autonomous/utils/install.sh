#!/bin/bash
# CARLA自动驾驶ROS包安装脚本

echo "安装CARLA自动驾驶ROS包..."

# 检查ROS环境
if [ -z "$ROS_DISTRO" ]; then
    echo "错误: ROS环境未设置"
    exit 1
fi

# 创建工作空间
mkdir -p ~/carla_ros_ws/src
cd ~/carla_ros_ws

# 复制文件
cp -r /path/to/your/carla_project/* src/

# 安装依赖
rosdep install --from-paths src --ignore-src -r -y

# 编译
catkin_make

# 添加环境变量
echo "source ~/carla_ros_ws/devel/setup.bash" >> ~/.bashrc

echo "安装完成！"
echo "请先启动CARLA服务器，然后运行: roslaunch carla_autonomous carla_autonomous.launch"