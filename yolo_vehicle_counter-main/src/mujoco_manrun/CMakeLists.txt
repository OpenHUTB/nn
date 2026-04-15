cmake_minimum_required(VERSION 3.0.2)
project(mujoco_manrun)

if($ENV{ROS_DISTRO} STREQUAL "noetic")
  set(PYTHON_EXECUTABLE /usr/bin/python3)
else()
  set(PYTHON_EXECUTABLE /usr/bin/python2)
endif()

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  rospy
  sensor_msgs
  std_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS 
    geometry_msgs 
    rospy 
    sensor_msgs 
    std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(DIRECTORY scripts/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
  FILES_MATCHING PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
)

catkin_install_python(PROGRAMS 
  scripts/main.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY models/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

if(CATKIN_ENABLE_TESTING)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/")
    install(DIRECTORY test/
      DESTINATION ${CATKIN_PACKAGE_TEST_DESTINATION}
      FILES_MATCHING PATTERN "*.py"
    )
    catkin_add_nosetests(test)
  endif()
endif()
