# --- START OF FILE cvips_utils.py ---
import numpy as np
import math
import carla

def build_projection_matrix(w, h, fov):
    """
    构建相机内参矩阵 (Intrinsic Matrix) K
    """
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_matrix(transform):
    """标准 CARLA 变换矩阵构建"""
    rotation = transform.rotation
    location = transform.location
    c_y = math.cos(math.radians(rotation.yaw))
    s_y = math.sin(math.radians(rotation.yaw))
    c_r = math.cos(math.radians(rotation.roll))
    s_r = math.sin(math.radians(rotation.roll))
    c_p = math.cos(math.radians(rotation.pitch))
    s_p = math.sin(math.radians(rotation.pitch))
    matrix = np.matrix(np.identity(4))
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
    return np.array(matrix)

def build_world_to_camera_matrix(camera_transform):
    """
    构建 [世界坐标] -> [相机坐标(OpenCV格式)] 的变换矩阵 (Extrinsic)
    关键修正：UE4 (X前,Y右,Z上) -> OpenCV (Z前,X右,Y下)
    """
    # 1. 计算相机在世界坐标下的位姿矩阵 (World -> CameraActor)
    cam_to_world = get_matrix(camera_transform)
    
    # 2. 求逆，得到 (CameraActor -> World) ... 实际上我们需要 World -> CameraActor
    # 也就是将世界点变换到相机原本的坐标系中
    world_to_cam_ue = np.linalg.inv(cam_to_world)
    
    # 3. 坐标轴修正矩阵 (UE4 -> OpenCV Standard)
    # UE4: X(前), Y(右), Z(上)
    # Cam: X(右), Y(下), Z(前)
    # 变换关系:
    # Cam_X = UE_Y
    # Cam_Y = -UE_Z
    # Cam_Z = UE_X
    calibration = np.zeros((4, 4))
    calibration[0, 1] = 1  # Row 0 (Out X) takes Col 1 (In Y)
    calibration[1, 2] = -1 # Row 1 (Out Y) takes Col 2 (In Z) * -1
    calibration[2, 0] = 1  # Row 2 (Out Z) takes Col 0 (In X)
    calibration[3, 3] = 1
    
    # 4. 组合矩阵: 先进行世界到UE相机的平移旋转，再进行轴变换
    w2c = np.dot(calibration, world_to_cam_ue)
    return w2c

def get_image_point(loc, K, w2c):
    """
    3D点 -> 2D像素
    """
    # [x, y, z, 1]
    point = np.array([loc.x, loc.y, loc.z, 1])
    
    # World -> Camera
    point_camera = np.dot(w2c, point)
    
    # 深度裁剪 (在相机背后的点剔除)
    if point_camera[2] <= 0:
        return None

    # Camera -> Image Plane (归一化)
    # [u*z, v*z, z] = K * [xc, yc, zc]
    point_img = np.dot(K, point_camera[:3])
    
    # 归一化
    u = point_img[0] / point_img[2]
    v = point_img[1] / point_img[2]
    
    return [int(u), int(v)]