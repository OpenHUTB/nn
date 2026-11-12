"""Launch file for teleoperation (ROS 2 Humble)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    scene = LaunchConfiguration("scene")
    return LaunchDescription(
        [
            DeclareLaunchArgument("scene", default_value="day"),
            Node(
                package="car_nn_sim",
                executable="sim_node",
                name="car_nn_sim",
                output="screen",
                parameters=[{"scene": scene, "policy": "", "rate": 20.0}],
            ),
            Node(
                package="teleop_twist_keyboard",
                executable="teleop_twist_keyboard",
                name="teleop",
                output="screen",
                remappings=[("/cmd_vel", "/car/cmd_vel")],
            ),
        ]
    )
