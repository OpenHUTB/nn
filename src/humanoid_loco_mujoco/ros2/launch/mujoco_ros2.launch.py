from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mujoco_ros2', 
            executable='mujoco_ros2_sim',  
            name='mujoco_ros2_sim',
            output='screen',
            parameters=[],  
        ),

        Node(
            package='mujoco_ros2', 
            executable='robot_controller', 
            name='robot_controller',
            output='screen',
            parameters=[], 
        )
    ])