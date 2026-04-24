"""
ROS特定的配置参数
"""

# ROS主题名称
ROS_TOPICS = {
    'vehicle_state': '/carla/vehicle_state',
    'vehicle_control': '/carla/vehicle_control',
    'camera_image': '/carla/camera/image',
    'segmentation_image': '/carla/camera/segmentation',
    'planned_path': '/carla/planned_path',
    'visualization': '/carla/visualization',
    'status': '/carla/status',
    'reward': '/carla/reward'
}

# ROS服务名称
ROS_SERVICES = {
    'reset': '/carla/reset',
    'start_episode': '/carla/start_episode',
    'stop': '/carla/stop'
}

# ROS参数
ROS_PARAMS = {
    'node_name': 'carla_autonomous_node',
    'rate_hz': 30.0,
    'queue_size': 10,
    'latch': False
}

# 坐标框架
ROS_FRAMES = {
    'map': 'map',
    'base_link': 'base_link',
    'camera': 'camera_link',
    'world': 'world'
}