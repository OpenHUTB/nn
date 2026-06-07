#!/usr/bin/env python3
"""
完整SLAM系统启动文件
同时启动所有节点
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os

def generate_launch_description():
    # 获取包路径
    package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 定义节点
    nodes = [
        # SLAM演示节点
        Node(
            package='slam_navigation',
            executable='slam_demo',
            name='slam_demo',
            output='screen',
            parameters=[{
                'use_sim_time': False,
            }]
        ),
        
        # 传感器模拟器节点
        Node(
            package='slam_navigation',
            executable='sensor_sim',
            name='sensor_simulator',
            output='screen',
            parameters=[{
                'use_sim_time': False,
            }]
        ),
        
        # RViz2可视化
        ExecuteProcess(
            cmd=['rviz2', '-d', os.path.join(package_path, 'rviz', 'slam_navigation.rviz')],
            output='screen'
        ),
    ]
    
    return LaunchDescription(nodes)
