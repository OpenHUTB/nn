cmake_minimum_required(VERSION 2.8.3)
project(mujoco_ros)

## 支持 C++11
add_compile_options(-std=c++11)

## 查找 ROS 1 依赖
find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  std_msgs
)

## 声明 catkin 包
catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs geometry_msgs std_msgs
)

## 包含路径
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## 安装 Python 脚本（main.py 用绝对路径，彻底避免歧义）
catkin_install_python(
  PROGRAMS
    scripts/publisher.py
    scripts/subscriber.py
    /home/lan/桌面/nn/src/Neuro_Mujoco/main.py  # 绝对路径！绝对路径！绝对路径！
  DESTINATION
    \${CATKIN_PACKAGE_BIN_DESTINATION}
)

## 安装 launch 文件
install(
  DIRECTORY launch/
  DESTINATION \${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)