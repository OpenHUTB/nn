# 基于ROS 2 Humble基础镜像
FROM osrf/ros:humble-desktop-full

# 设置非交互模式，避免tty问题
ENV DEBIAN_FRONTEND=noninteractive

# 设置工作目录
WORKDIR /ros2_ws

# 安装依赖
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖（指定兼容版本）
RUN pip3 install \
    carla==0.9.14 \
    psutil==5.9.5 \
    PyYAML==6.0.1 \
    tf-transformations==2.0.0

# 复制源码到容器
COPY ./src /ros2_ws/src

# 构建ROS 2包
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

# 设置环境变量（自动加载ROS环境）
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc

# 健康检查（检查节点是否正常运行）
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ros2 node list | grep -q "carla_global_planner_node" || exit 1

# 容器启动命令
CMD ["bash", "-c", "source /ros2_ws/install/setup.bash && \
    ros2 run carla_global_planner carla_global_planner_node"]
