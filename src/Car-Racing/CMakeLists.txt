cmake_minimum_required(VERSION 2.8.3)
project(car_racing_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# 标记脚本为可执行
catkin_install_python(PROGRAMS scripts/car_racing_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
