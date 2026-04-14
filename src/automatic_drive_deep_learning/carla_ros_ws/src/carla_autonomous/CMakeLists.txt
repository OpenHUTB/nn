cmake_minimum_required(VERSION 3.0.2)
project(carla_autonomous)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  visualization_msgs
  tf
  cv_bridge
  message_generation
)

add_service_files(
  FILES
  Reset.srv
  StartEpisode.srv
  Stop.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS 
  rospy 
  std_msgs 
  geometry_msgs 
  sensor_msgs 
  nav_msgs 
  visualization_msgs 
  tf 
  cv_bridge
  message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# 安装Python脚本
catkin_install_python(PROGRAMS
  scripts/carla_autonomous_node.py
  scripts/carla_control_client.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 安装其他文件
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

install(DIRECTORY src/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
  PATTERN "*.py"
)

install(DIRECTORY models/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models
)

install(DIRECTORY pics/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/pics
)

install(DIRECTORY utils/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/utils
)