import numpy as np
import math
import carla

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_matrix(transform):
    """Local to World 4x4 Matrix"""
    rotation = transform.rotation
    location = transform.location
    c_y = math.cos(math.radians(rotation.yaw))
    s_y = math.sin(math.radians(rotation.yaw))
    c_r = math.cos(math.radians(rotation.roll))
    s_r = math.sin(math.radians(rotation.roll))
    c_p = math.cos(math.radians(rotation.pitch))
    s_p = math.sin(math.radians(rotation.pitch))
    matrix = np.identity(4)
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = c_p * s_y
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix

def build_world_to_camera_matrix(camera_transform):
    """World to Camera 4x4 Matrix (OpenCV format)"""
    cam_to_world = get_matrix(camera_transform)
    world_to_cam_ue = np.linalg.inv(cam_to_world)
    # 坐标轴转换：UE4 -> OpenCV
    calibration = np.array([
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
    ])
    return np.dot(calibration, world_to_cam_ue)

def get_image_point(loc, K, w2c):
    """3D Loc -> 2D Pixel"""
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    if point_camera[2] <= 0: return None
    point_img = np.dot(K, point_camera[:3])
    u = point_img[0] / point_img[2]
    v = point_img[1] / point_img[2]
    return [int(u), int(v)]