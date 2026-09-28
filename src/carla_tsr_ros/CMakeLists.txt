cmake_minimum_required(VERSION 3.0.2)
project(carla_tsr_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  geometry_msgs
  cv_bridge
)

catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs std_msgs geometry_msgs cv_bridge
)

catkin_install_python(PROGRAMS
  ros_nodes/image_publisher.py
  ros_nodes/perception_node.py
  ros_nodes/control_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
