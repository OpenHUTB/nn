import numpy as np
import carla

def calculate_distance(location1, location2):
    """计算两个Carla Location的距离"""
    return np.linalg.norm([
        location1.x - location2.x,
        location1.y - location2.y,
        location1.z - location2.z
    ])

def transform_to_world(vehicle_transform, local_point):
    """将车辆局部坐标转换为世界坐标"""
    local_point = carla.Vector3D(*local_point)
    world_point = vehicle_transform.transform(local_point)
    return world_point

def get_lane_center(waypoint):
    """获取车道中心线"""
    return waypoint.transform.location + carla.Location(z=0.1)