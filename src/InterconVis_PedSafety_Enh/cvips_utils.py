import numpy as np
import math
import carla

def build_projection_matrix(w, h, fov):
    """
    构建相机的内参矩阵 (Intrinsic Matrix)
    """
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    """
    将 3D 世界坐标点 (loc) 投影到 2D 像素坐标 (u, v)
    :param loc: carla.Location / carla.Vector3D
    :param K: 内参矩阵 (3x3)
    :param w2c: 世界到相机的变换矩阵 (4x4)
    """
    # 1. 构建齐次坐标 [x, y, z, 1]
    point = np.array([loc.x, loc.y, loc.z, 1])
    
    # 2. 变换到相机坐标系 (Camera Coordinate)
    point_camera = np.dot(w2c, point)
    
    # 3. 归一化相机平面 (除去深度 Z) [y, -z, x] -> standard [x, y, z] conversion happening inside w2c logic usually
    # 但在 CARLA helper 中，我们直接用投影公式:
    # 这里的 w2c 实际上已经包含了坐标轴的旋转 (UE4 -> OpenCV)
    
    # [x,y,z] -> [X_img, Y_img, 1] * depth
    point_img = np.dot(K, point_camera[:3])
    
    # 4. 归一化获取像素坐标
    pixel = [int(point_img[0] / point_img[2]), int(point_img[1] / point_img[2])]
    return pixel

def get_matrix(transform):
    """
    将 CARLA 的 transform 对象转换为 4x4 变换矩阵
    """
    yaw = math.radians(transform.rotation.yaw)
    pitch = math.radians(transform.rotation.pitch)
    roll = math.radians(transform.rotation.roll)

    c_y = math.cos(yaw)
    s_y = math.sin(yaw)
    c_p = math.cos(pitch)
    s_p = math.sin(pitch)
    c_r = math.cos(roll)
    s_r = math.sin(roll)

    matrix = np.identity(4)
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = c_p * s_y
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    matrix[0, 3] = transform.location.x
    matrix[1, 3] = transform.location.y
    matrix[2, 3] = transform.location.z
    return matrix

def build_world_to_camera_matrix(camera_transform):
    """
    构建 世界坐标系 -> 相机像素坐标系 的外参矩阵 (Extrinsic Matrix)
    关键：包含坐标轴置换 (UE4 -> Standard CV)
    """
    # 1. 获取相机在世界坐标系的位姿矩阵 M
    cam_M = get_matrix(camera_transform)
    
    # 2. 求逆，得到 世界->相机 的变换
    w2c = np.linalg.inv(cam_M)
    
    # 3. 坐标轴修正矩阵 (UE4 x-forward, y-right, z-up -> Camera z-forward, x-right, y-down)
    calibration = np.identity(4)
    calibration[0, 2] = 1  # x -> z
    calibration[1, 0] = -1 # y -> -x
    calibration[2, 1] = -1 # z -> -y
    
    # 组合
    w2c = np.dot(calibration, w2c)
    return w2c