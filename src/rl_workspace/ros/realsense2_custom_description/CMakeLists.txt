cmake_minimum_required(VERSION 3.0.2)
project(realsense2_custom_description)

find_package(catkin REQUIRED COMPONENTS)
catkin_package(CATKIN_DEPENDS)

install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

include_directories(${catkin_INCLUDE_DIRS})

if(CATKIN_ENABLE_TESTING)
    find_package(rostest REQUIRED)
    # for each test in the test directory add it as a test
    file(GLOB TEST_LAUNCH_FILES tests/*.test)
    foreach(TEST_LAUNCH_FILE ${TEST_LAUNCH_FILES})
        add_rostest(${TEST_LAUNCH_FILE})
    endforeach()
endif()
