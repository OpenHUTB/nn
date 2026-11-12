"""Launch file for neural-network driving (ROS 2 Humble)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    scene = LaunchConfiguration("scene")
    policy = LaunchConfiguration("policy")
    rate = LaunchConfiguration("rate")
    return LaunchDescription(
        [
            DeclareLaunchArgument("scene", default_value="day", description="day | dusk | fog"),
            DeclareLaunchArgument(
                "policy",
                default_value="",
                description="path to a .pt checkpoint; empty means teleoperation",
            ),
            DeclareLaunchArgument("rate", default_value="20.0"),
            Node(
                package="car_nn_sim",
                executable="sim_node",
                name="car_nn_sim",
                output="screen",
                parameters=[{"scene": scene, "policy": policy, "rate": rate}],
            ),
        ]
    )
