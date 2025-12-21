#!/bin/bash
# 启动CARLA服务器和ROS节点

# 启动CARLA服务器
echo "启动CARLA服务器..."
cd ~/CARLA_0.9.13
./CarlaUE4.sh -quality-level=Low -fps=20 &

# 等待CARLA启动
sleep 10

# 设置ROS环境
source ~/carla_ros_ws/devel/setup.bash

# 启动ROS节点
echo "启动ROS节点..."
roslaunch carla_autonomous carla_autonomous.launch

# 清理
echo "正在关闭..."
pkill CarlaUE4