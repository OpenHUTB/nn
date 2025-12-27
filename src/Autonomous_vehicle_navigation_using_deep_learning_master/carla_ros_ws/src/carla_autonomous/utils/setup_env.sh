#!/bin/bash
# 设置环境变量

export CARLA_ROOT=/path/to/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export ROS_MASTER_URI=http://localhost:11311
export ROS_IP=localhost

# 添加工作空间
source ~/carla_ros_ws/devel/setup.bash

echo "环境变量设置完成"