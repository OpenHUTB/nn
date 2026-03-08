cmake_minimum_required(VERSION 3.0.2)
# 功能包名称（必须与package.xml中的name一致）
project(mmap_test)

# -------------------------- 适配Python3.7.5 --------------------------
# 指定你的虚拟环境Python解释器路径（关键！）
set(PYTHON_EXECUTABLE /home/pan-j-l/my_ros_project/venv_py37/bin/python3)
# 系统Python3.7的头文件路径（Ubuntu20.04默认路径，一般无需修改）
set(PYTHON_INCLUDE_DIR /usr/include/python3.7m)
# 系统Python3.7的库文件路径（Ubuntu20.04默认路径，一般无需修改）
set(PYTHON_LIBRARY /usr/lib/x86_64-linux-gnu/libpython3.7m.so)

# -------------------------- 查找ROS依赖包 --------------------------
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
)

# -------------------------- 声明Catkin包 --------------------------
catkin_package(
  # 声明该包依赖的其他ROS包，供其他包引用
  CATKIN_DEPENDS rospy std_msgs sensor_msgs cv_bridge
)

# -------------------------- 安装Python脚本 --------------------------
# 将scripts目录下的Python脚本安装到ROS的devel目录，实现全局调用
catkin_install_python(PROGRAMS
  scripts/ros_test_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# -------------------------- 包含目录 --------------------------
include_directories(
  ${catkin_INCLUDE_DIRS}
)