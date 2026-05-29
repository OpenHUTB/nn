cmake_minimum_required(VERSION 3.0.2)
project(img2vec)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
)

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  ImgEmbedding.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
)


###################################
## catkin specific configuration ##
###################################

catkin_package(

CATKIN_DEPENDS message_runtime 
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)

 catkin_install_python(PROGRAMS
   scripts/embeddings.py
   scripts/imgPublisher.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )


