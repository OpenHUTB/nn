cmake_minimum_required(VERSION 3.0.2)
project(handy_environment)

find_package(catkin REQUIRED COMPONENTS)
catkin_package(CATKIN_DEPENDS)

install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

file(GLOB SCRIPT_FILES script/*)
install(
  PROGRAMS ${SCRIPT_FILES}
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

include_directories(${catkin_INCLUDE_DIRS})

if(CATKIN_ENABLE_TESTING)
    find_package(rostest REQUIRED)
    file(GLOB TEST_LAUNCH_FILES tests/*.test)
    foreach(TEST_LAUNCH_FILE ${TEST_LAUNCH_FILES})
        add_rostest(${TEST_LAUNCH_FILE})
    endforeach()
endif()
