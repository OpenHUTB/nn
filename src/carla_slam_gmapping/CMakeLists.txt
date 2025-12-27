cmake_minimum_required(VERSION 3.0.2)
project(carla_slam_gmapping)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  move_base_msgs
  visualization_msgs
  carla_msgs
  tf
  tf2_ros
  cv_bridge
  image_transport
)

## System dependencies are found with CMake's conventions
# find_package(OpenCV REQUIRED)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS 
    rospy
    roscpp
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    move_base_msgs
    visualization_msgs
    carla_msgs
    tf
    tf2_ros
    cv_bridge
    image_transport
)

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

## Mark Python scripts as executable
catkin_install_python(PROGRAMS
  goal_publisher.py
  main.py
  scenario_manager.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark launch files for installation
install(FILES
  carla_navigation.launch
  carla_slam_gmapping.launch
  carla_slam_navigation.launch
  save_map.launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

## Mark config files for installation
install(FILES
  costmap_common_params.yaml
  dwa_local_planner_params.yaml
  global_costmap_params.yaml
  local_costmap_params.yaml
  move_base_params.yaml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

## Mark shell scripts for installation
install(PROGRAMS
  start_slam_navigation.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark other files for installation (e.g. documentation)
install(FILES
  README.md
  requirements.txt
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_carla_slam_gmapping.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
