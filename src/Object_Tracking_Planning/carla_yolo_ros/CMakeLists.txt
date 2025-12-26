cmake_minimum_required(VERSION 3.0.2)
project(carla_yolo_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  cv_bridge
  image_transport
  tf
)

catkin_package(
  CATKIN_DEPENDS 
  roscpp 
  rospy 
  std_msgs 
  sensor_msgs 
  geometry_msgs 
  nav_msgs 
  visualization_msgs 
  cv_bridge 
  image_transport 
  tf
)

catkin_install_python(PROGRAMS
  scripts/carla_ros_bridge.py
  scripts/test_image_publisher.py
  scripts/teleop_keyboard.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  launch
  config
  models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)