cmake_minimum_required(VERSION 3.0.2)
project(carla_tracking_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  cv_bridge
  image_transport
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs cv_bridge image_transport
)

install(DIRECTORY launch/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

install(FILES requirements.txt
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
