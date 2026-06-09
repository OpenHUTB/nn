# realsense2_description

This package contains a modified version of urdf files for the realsense2 camera.
The original files can be found in the [rickstaa/realsense-ros-gazebo](https://github.com/rickstaa/realsense-ros-gazebo) repository.

The main difference in this model is that we can turn off gravity for the camera link.
This is useful for simulating the camera in a fixed position.
Use the `realsense2_description` package for other files such as the rviz config.
The included launch file will configure the topics and additional align the depth image to the color image.
