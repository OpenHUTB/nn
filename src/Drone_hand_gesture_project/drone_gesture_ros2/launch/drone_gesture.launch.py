from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 手势控制节点
        Node(
            package='drone_gesture_control',
            executable='gesture_control_node',
            name='gesture_control_node',
            output='screen',
            parameters=[
                {'simulation_mode': True},
                {'camera_id': 1},
                {'ml_model_path': 'dataset/models/gesture_svm.pkl'}
            ]
        ),
        
        # 可视化节点
        Node(
            package='drone_gesture_control',
            executable='visualization_node',
            name='visualization_node',
            output='screen'
        ),
        
        # 可选的：状态监控节点
        Node(
            package='drone_gesture_control',
            executable='status_monitor_node',
            name='status_monitor_node',
            output='screen'
        )
    ])
