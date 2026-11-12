cmake_minimum_required(VERSION 3.0.2)
project(uav_flight_base)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  nav_msgs
  tf2_ros
)

catkin_package(
  CATKIN_DEPENDS rospy geometry_msgs nav_msgs tf2_ros
)

catkin_install_python(PROGRAMS
  scripts/uav_teleop_key.py
  scripts/uav_mock_sim.py
  scripts/uav_state_monitor.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
