"""
自动驾驶系统配置文件
"""

# Carla连接配置
CARLA_SERVER = 'localhost'
CARLA_PORT = 2000
WORLD_NAME = 'Town05'

# 显示配置
PYGAME_WIDTH = 1280
PYGAME_HEIGHT = 720
PYGAME_FPS = 30
PYGAME_FOV = 90

# 车辆控制配置
VEHICLE_CONFIG = {
    'MAX_SPEED_KMH': 60,        # 最大速度
    'MIN_SPEED_KMH': 20,        # 最小速度
    'DEFAULT_THROTTLE': 0.6,    # 默认油门
    'MAX_STEER': 0.7,           # 最大转向
    'MAX_BRAKE': 1.0,           # 最大刹车
}

# MPC控制器配置
MPC_CONFIG = {
    'PREDICTION_HORIZON': 10,   # 预测时域
    'TIME_STEP': 0.1,           # 时间步长
    'WHEELBASE': 3.0,           # 轴距
    'MAX_STEER_RATE': 0.1,      # 最大转向变化率
}

# 弯道控制配置
CURVE_CONFIG = {
    'LOOKAHEAD_POINTS': 8,      # 前瞻点数
    'CURVATURE_THRESHOLD': 0.3, # 弯道阈值
    'SMOOTHING_FACTOR': 0.3,    # 平滑系数
}

# 车道保持配置
LANE_KEEP_CONFIG = {
    'ACTIVE': True,             # 是否激活
    'KP': 0.05,                 # 比例系数
    'MAX_CORRECTION': 0.15,     # 最大修正
    'WARNING_THRESHOLD': 1.0,   # 警告阈值
}

# 性能配置
PERFORMANCE_CONFIG = {
    'ENABLE_FPS_DISPLAY': True, # 显示FPS
    'LOG_INTERVAL': 100,        # 日志间隔
    'DEBUG_MODE': False,        # 调试模式
}