"""Launch file for SLAM mapping (ROS 2 Humble)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    scene = LaunchConfiguration("scene")
    policy = LaunchConfiguration("policy")
    return LaunchDescription(
        [
            DeclareLaunchArgument("scene", default_value="day"),
            DeclareLaunchArgument("policy", default_value=""),
            Node(
                package="car_nn_sim",
                executable="sim_node",
                name="car_nn_sim",
                output="screen",
                parameters=[{"scene": scene, "policy": policy, "rate": 20.0}],
            ),
            Node(package="car_nn_sim", executable="slam_node", name="car_slam", output="screen"),
        ]
    )
