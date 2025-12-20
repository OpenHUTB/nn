# config.py
# CARLA 连接配置
CARLA_SERVER = 'localhost'  # CARLA 服务器地址
WORLD_NAME = 'Town05'       # 默认地图名称

# PyGame 显示配置
PYGAME_WIDTH = 800          # 窗口宽度
PYGAME_HEIGHT = 600         # 窗口高度
PYGAME_FPS = 20             # 帧率
PYGAME_FOV = 90             # 相机视场角（Field of View）

# 相机配置
CAMERA_CONFIG = {
    'image_size_x': PYGAME_WIDTH,
    'image_size_y': PYGAME_HEIGHT,
    'fov': PYGAME_FOV,
}

# MPC 控制参数
MPC_CONFIG = {
    'N': 10,                # 预测时域
    'DT': 0.1,              # 时间步长
    'L_F': 1.5,             # 前轴到质心距离
    'L_R': 1.5,             # 后轴到质心距离
    'MAX_STEER': 0.6,       # 最大转向角
    'MAX_DSTEER': 0.1,      # 最大转向角变化率
}

# 弯道减速参数
CURVE_CONTROL = {
    'MAX_SPEED_KMH': 60,    # 直道最大速度（km/h）
    'MIN_SPEED_KMH': 20,    # 弯道最小速度（km/h）
    'DEFAULT_THROTTLE': 0.6, # 默认油门值
    'CURVE_THRESHOLD': 0.3, # 弯道阈值
    'LOOKAHEAD_DISTANCE': 8, # 前瞻距离（点数量）
    'SMOOTHING_ALPHA': 0.3,  # 平滑系数
}

# 感知减速参数（如果还需要）
PERCEPTION = {
    'SAFETY_DISTANCE': 10.0,  # 安全距离（米）
    'EMERGENCY_DISTANCE': 5.0,  # 紧急刹车距离（米）
    'DETECTION_RANGE': 30.0,  # 检测范围（米）
    'DETECTION_ANGLE': 60,    # 检测角度（度）
    'DEFAULT_THROTTLE': 0.6,  # 默认油门值
}

# 车辆参数
VEHICLE_CONFIG = {
    'MAX_SPEED': 70,         # 最大速度 km/h
    'MAX_THROTTLE': 0.8,     # 最大油门
    'MAX_BRAKE': 1.0,        # 最大刹车
    'MAX_STEER': 0.7,        # 最大转向
}