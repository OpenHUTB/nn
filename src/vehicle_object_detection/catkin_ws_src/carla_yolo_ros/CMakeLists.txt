cmake_minimum_required(VERSION 2.8.3)
project(carla_yolo_ros)

## Compile as C++11 (Standard for ROS Kinetic)
add_compile_options(-std=c++11)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
)

## Catkin specific configuration
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES carla_yolo_ros
  CATKIN_DEPENDS rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)

## Build configuration
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Install rules (Ensure Python scripts are executable)
catkin_install_python(PROGRAMS
  src/object_detection_node.py
  src/traffic_generator_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)