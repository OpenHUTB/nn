cmake_minimum_required(VERSION 3.0.2)
project(carla_rl_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/ros_eval_node.py
  scripts/ros_train_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
