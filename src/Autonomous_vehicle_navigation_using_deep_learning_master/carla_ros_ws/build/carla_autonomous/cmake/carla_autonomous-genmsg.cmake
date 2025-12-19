# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "carla_autonomous: 0 messages, 3 services")

set(MSG_I_FLAGS "-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg;-Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(carla_autonomous_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/yume/carla_ros_ws/src/carla_autonomous/srv/Reset.srv" NAME_WE)
add_custom_target(_carla_autonomous_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "carla_autonomous" "/home/yume/carla_ros_ws/src/carla_autonomous/srv/Reset.srv" "std_msgs/MultiArrayDimension:std_msgs/Float32MultiArray:std_msgs/MultiArrayLayout"
)

get_filename_component(_filename "/home/yume/carla_ros_ws/src/carla_autonomous/srv/StartEpisode.srv" NAME_WE)
add_custom_target(_carla_autonomous_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "carla_autonomous" "/home/yume/carla_ros_ws/src/carla_autonomous/srv/StartEpisode.srv" ""
)

get_filename_component(_filename "/home/yume/carla_ros_ws/src/carla_autonomous/srv/Stop.srv" NAME_WE)
add_custom_target(_carla_autonomous_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "carla_autonomous" "/home/yume/carla_ros_ws/src/carla_autonomous/srv/Stop.srv" ""
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages

### Generating Services
_generate_srv_cpp(carla_autonomous
  "/home/yume/carla_ros_ws/src/carla_autonomous/srv/Reset.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Float32MultiArray.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/carla_autonomous
)
_generate_srv_cpp(carla_autonomous
  "/home/yume/carla_ros_ws/src/carla_autonomous/srv/StartEpisode.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/carla_autonomous
)
_generate_srv_cpp(carla_autonomous
  "/home/yume/carla_ros_ws/src/carla_autonomous/srv/Stop.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/carla_autonomous
)

### Generating Module File
_generate_module_cpp(carla_autonomous
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/carla_autonomous
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(carla_autonomous_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(carla_autonomous_generate_messages carla_autonomous_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/yume/carla_ros_ws/src/carla_autonomous/srv/Reset.srv" NAME_WE)
add_dependencies(carla_autonomous_generate_messages_cpp _carla_autonomous_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/yume/carla_ros_ws/src/carla_autonomous/srv/StartEpisode.srv" NAME_WE)
add_dependencies(carla_autonomous_generate_messages_cpp _carla_autonomous_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/yume/carla_ros_ws/src/carla_autonomous/srv/Stop.srv" NAME_WE)
add_dependencies(carla_autonomous_generate_messages_cpp _carla_autonomous_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(carla_autonomous_gencpp)
add_dependencies(carla_autonomous_gencpp carla_autonomous_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS carla_autonomous_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages

### Generating Services
_generate_srv_eus(carla_autonomous
  "/home/yume/carla_ros_ws/src/carla_autonomous/srv/Reset.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Float32MultiArray.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/carla_autonomous
)
_generate_srv_eus(carla_autonomous
  "/home/yume/carla_ros_ws/src/carla_autonomous/srv/StartEpisode.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/carla_autonomous
)
_generate_srv_eus(carla_autonomous
  "/home/yume/carla_ros_ws/src/carla_autonomous/srv/Stop.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/carla_autonomous
)

### Generating Module File
_generate_module_eus(carla_autonomous
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/carla_autonomous
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(carla_autonomous_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(carla_autonomous_generate_messages carla_autonomous_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/yume/carla_ros_ws/src/carla_autonomous/srv/Reset.srv" NAME_WE)
add_dependencies(carla_autonomous_generate_messages_eus _carla_autonomous_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/yume/carla_ros_ws/src/carla_autonomous/srv/StartEpisode.srv" NAME_WE)
add_dependencies(carla_autonomous_generate_messages_eus _carla_autonomous_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/yume/carla_ros_ws/src/carla_autonomous/srv/Stop.srv" NAME_WE)
add_dependencies(carla_autonomous_generate_messages_eus _carla_autonomous_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(carla_autonomous_geneus)
add_dependencies(carla_autonomous_geneus carla_autonomous_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS carla_autonomous_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages

### Generating Services
_generate_srv_lisp(carla_autonomous
  "/home/yume/carla_ros_ws/src/carla_autonomous/srv/Reset.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Float32MultiArray.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/carla_autonomous
)
_generate_srv_lisp(carla_autonomous
  "/home/yume/carla_ros_ws/src/carla_autonomous/srv/StartEpisode.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/carla_autonomous
)
_generate_srv_lisp(carla_autonomous
  "/home/yume/carla_ros_ws/src/carla_autonomous/srv/Stop.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/carla_autonomous
)

### Generating Module File
_generate_module_lisp(carla_autonomous
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/carla_autonomous
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(carla_autonomous_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(carla_autonomous_generate_messages carla_autonomous_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/yume/carla_ros_ws/src/carla_autonomous/srv/Reset.srv" NAME_WE)
add_dependencies(carla_autonomous_generate_messages_lisp _carla_autonomous_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/yume/carla_ros_ws/src/carla_autonomous/srv/StartEpisode.srv" NAME_WE)
add_dependencies(carla_autonomous_generate_messages_lisp _carla_autonomous_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/yume/carla_ros_ws/src/carla_autonomous/srv/Stop.srv" NAME_WE)
add_dependencies(carla_autonomous_generate_messages_lisp _carla_autonomous_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(carla_autonomous_genlisp)
add_dependencies(carla_autonomous_genlisp carla_autonomous_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS carla_autonomous_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages

### Generating Services
_generate_srv_nodejs(carla_autonomous
  "/home/yume/carla_ros_ws/src/carla_autonomous/srv/Reset.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Float32MultiArray.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/carla_autonomous
)
_generate_srv_nodejs(carla_autonomous
  "/home/yume/carla_ros_ws/src/carla_autonomous/srv/StartEpisode.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/carla_autonomous
)
_generate_srv_nodejs(carla_autonomous
  "/home/yume/carla_ros_ws/src/carla_autonomous/srv/Stop.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/carla_autonomous
)

### Generating Module File
_generate_module_nodejs(carla_autonomous
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/carla_autonomous
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(carla_autonomous_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(carla_autonomous_generate_messages carla_autonomous_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/yume/carla_ros_ws/src/carla_autonomous/srv/Reset.srv" NAME_WE)
add_dependencies(carla_autonomous_generate_messages_nodejs _carla_autonomous_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/yume/carla_ros_ws/src/carla_autonomous/srv/StartEpisode.srv" NAME_WE)
add_dependencies(carla_autonomous_generate_messages_nodejs _carla_autonomous_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/yume/carla_ros_ws/src/carla_autonomous/srv/Stop.srv" NAME_WE)
add_dependencies(carla_autonomous_generate_messages_nodejs _carla_autonomous_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(carla_autonomous_gennodejs)
add_dependencies(carla_autonomous_gennodejs carla_autonomous_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS carla_autonomous_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages

### Generating Services
_generate_srv_py(carla_autonomous
  "/home/yume/carla_ros_ws/src/carla_autonomous/srv/Reset.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayDimension.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Float32MultiArray.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/MultiArrayLayout.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/carla_autonomous
)
_generate_srv_py(carla_autonomous
  "/home/yume/carla_ros_ws/src/carla_autonomous/srv/StartEpisode.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/carla_autonomous
)
_generate_srv_py(carla_autonomous
  "/home/yume/carla_ros_ws/src/carla_autonomous/srv/Stop.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/carla_autonomous
)

### Generating Module File
_generate_module_py(carla_autonomous
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/carla_autonomous
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(carla_autonomous_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(carla_autonomous_generate_messages carla_autonomous_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/yume/carla_ros_ws/src/carla_autonomous/srv/Reset.srv" NAME_WE)
add_dependencies(carla_autonomous_generate_messages_py _carla_autonomous_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/yume/carla_ros_ws/src/carla_autonomous/srv/StartEpisode.srv" NAME_WE)
add_dependencies(carla_autonomous_generate_messages_py _carla_autonomous_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/yume/carla_ros_ws/src/carla_autonomous/srv/Stop.srv" NAME_WE)
add_dependencies(carla_autonomous_generate_messages_py _carla_autonomous_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(carla_autonomous_genpy)
add_dependencies(carla_autonomous_genpy carla_autonomous_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS carla_autonomous_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/carla_autonomous)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/carla_autonomous
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(carla_autonomous_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(carla_autonomous_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/carla_autonomous)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/carla_autonomous
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(carla_autonomous_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(carla_autonomous_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/carla_autonomous)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/carla_autonomous
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(carla_autonomous_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(carla_autonomous_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/carla_autonomous)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/carla_autonomous
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(carla_autonomous_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(carla_autonomous_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/carla_autonomous)
  install(CODE "execute_process(COMMAND \"/home/yume/miniconda3/envs/Autonomous_env/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/carla_autonomous\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/carla_autonomous
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(carla_autonomous_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(carla_autonomous_generate_messages_py geometry_msgs_generate_messages_py)
endif()
