cmake_minimum_required(VERSION 3.8)
project(mujoco_ros_demo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


set(MUJOCO_INCLUDE_DIRS "/usr/local/mujoco-3.3.7/include")
set(MUJOCO_LIBRARY_DIRS "/usr/local/mujoco-3.3.7/lib")

# 包含MuJoCo头文件目录
include_directories(${MUJOCO_INCLUDE_DIRS})
# 链接MuJoCo库文件目录
link_directories(${MUJOCO_LIBRARY_DIRS})


find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ament_index_cpp REQUIRED)


# mujoco_publisher_cpp（MuJoCo仿真发布节点）
add_executable(mujoco_publisher_cpp mujoco_demo_cpp/mujoco_publisher.cpp)
ament_target_dependencies(mujoco_publisher_cpp
  rclcpp std_msgs Eigen3 ament_index_cpp)
target_link_libraries(mujoco_publisher_cpp mujoco)  # 链接MuJoCo库

# data_subscriber_cpp（基础数据订阅节点）
add_executable(data_subscriber_cpp mujoco_demo_cpp/data_subscriber.cpp)
ament_target_dependencies(data_subscriber_cpp rclcpp std_msgs)

# 安装C++节点
install(TARGETS
  mujoco_publisher_cpp
  data_subscriber_cpp
  DESTINATION lib/${PROJECT_NAME}
)

# 安装config（模型文件）
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()