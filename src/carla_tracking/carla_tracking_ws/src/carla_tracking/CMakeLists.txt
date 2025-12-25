cmake_minimum_required(VERSION 3.0.2)
project(carla_tracking)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  message_generation
  rospy
  sensor_msgs
  std_msgs
)

## 声明自定义消息
add_message_files(
  FILES
  TrackedObject.msg
)

## 生成消息
generate_messages(
  DEPENDENCIES
  std_msgs
)

## 声明catkin包
catkin_package(
  CATKIN_DEPENDS cv_bridge message_runtime rospy sensor_msgs std_msgs
)

## 包含目录
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## 标记Python脚本为可执行
catkin_install_python(PROGRAMS
  scripts/publisher.py
  scripts/subscriber.py
  scripts/carla_ros_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
