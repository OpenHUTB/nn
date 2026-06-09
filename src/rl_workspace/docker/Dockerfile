FROM ros:noetic-ros-core-focal

ENV DEBIAN_FRONTEND noninteractive
RUN apt update && \
    apt install -y --no-install-recommends \
    apt-utils \
    build-essential \
    git \
    python3-dev \
    python3-pip \
    python3-requests \
    python3.8-venv \
    python-is-python3 \
    python3-catkin-tools \
    python3-rosdep \
    wget \
    curl \
    && \
    apt clean

# pull any dependencies needed by the workspace. here's a command that might
# make it a bit easier to enumerate and paste, when run in a container that
# doesn't have any dependencies and the right packages in the workspace.
#
#   rosdep install --from-paths src --ignore-src -y --simulate | \
#   grep apt-get | tr -s " " | cut -d" " -f5 | \
#   xargs printf "%s \\ \n"
#
RUN apt update && \
    apt install -y --no-install-recommends \
    # dependencies we get from above
    ros-noetic-robot-state-publisher \
    ros-noetic-rviz \
    ros-noetic-joint-state-publisher \
    gazebo11 \
    ros-noetic-hardware-interface \
    ros-noetic-xacro \
    ros-noetic-moveit-ros-move-group \
    ros-noetic-moveit-fake-controller-manager \
    ros-noetic-moveit-simple-controller-manager \
    ros-noetic-moveit-kinematics \
    ros-noetic-moveit-planners-ompl \
    ros-noetic-moveit-ros-visualization \
    ros-noetic-cv-bridge \
    ros-noetic-ros-pytest \
    ros-noetic-control-msgs \
    python3-serial \
    ros-noetic-ros-control \
    ros-noetic-ros-controllers \
    ros-noetic-controller-manager \
    ros-noetic-joint-state-controller \
    ros-noetic-position-controllers \
    ros-noetic-joint-trajectory-controller \
    python-yaml \
    ros-noetic-gazebo-ros-control \
    ros-noetic-rqt-gui \
    ros-noetic-effort-controllers \
    libeigen3-dev \
    ros-noetic-image-transport \
    ros-noetic-tf \
    ros-noetic-ddynamic-reconfigure \
    ros-noetic-diagnostic-updater \
    ros-noetic-librealsense2 \
    ros-noetic-gazebo-ros \
    ros-noetic-camera-info-manager \
    ros-noetic-moveit-commander \
    ros-noetic-rospy-message-converter \
    # also include image_view for debugging, and usb_cam for a personal camera
    ros-noetic-image-view \
    ros-noetic-usb-cam \
    && \
    apt clean

# install node
RUN curl -fsSL https://deb.nodesource.com/setup_21.x | bash \
    && \
    apt install -y --no-install-recommends \
    nodejs \
    && \
    apt clean

RUN pip install --upgrade setuptools pip wheel cython

WORKDIR /catkin_ws
RUN mkdir src
RUN cd src && \
    git clone https://github.com/ivaROS/ivaDynamixel.git && \
    git clone https://github.com/ivaROS/ivaHandy.git && \
    git clone --recurse-submodules https://github.com/rickstaa/realsense-ros-gazebo.git && \
    git clone https://github.com/ivalab/simData.git

# add the current ros module and install the requirements
WORKDIR /catkin_ws/src/app
ENV PATH=/root/.local/bin:$PATH
ADD ./agent agent
ADD requirements.txt .
RUN pip install -r requirements.txt

ADD . ./

# build the workspace
WORKDIR /catkin_ws

RUN catkin config --init --extend /opt/ros/noetic && \
    rosdep init && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y && \
    catkin build

WORKDIR /catkin_ws/src/app
COPY ./docker/bin/ros_entrypoint.sh /
ENTRYPOINT ["/ros_entrypoint.sh",  "stdbuf", "-o", "L"]
# exit after starting
CMD ["true"]
