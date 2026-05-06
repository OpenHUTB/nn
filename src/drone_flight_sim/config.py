# config.py
"""无人机飞行配置参数模块

本模块定义了无人机飞行过程中使用的所有配置参数，
包括飞行参数、碰撞检测参数、降落参数等常量。
"""

# 从 typing 模块导入 List 和 Tuple 类型（保留用于未来扩展）
# from typing import List, Tuple


class FlightConfig:
    """无人机飞行配置类

    包含所有飞行相关的配置参数，使用类属性定义常量。
    """

    # ==================== 飞行参数 ====================
    # 起飞高度（负值表示向上，AirSim 中 Z 轴向下为正）
    TAKEOFF_HEIGHT = -5
    # 飞行速度，单位：米/秒
    FLIGHT_VELOCITY = 4
    # 最大飞行时间，单位：秒，超过此时间将强制结束飞行
    MAX_FLIGHT_TIME = 120

    # ==================== 碰撞检测参数 ====================
    # 碰撞冷却时间，单位：秒，用于防止重复触发碰撞检测
    COLLISION_COOLDOWN = 1.0
    # 地面判断阈值，单位：米，高度低于此值认为是地面接触
    GROUND_HEIGHT_THRESHOLD = 1.5
    # 到达目标点的容差，单位：米，用于判断是否已到达目标位置
    ARRIVAL_TOLERANCE = 1.5
    # 最大自动恢复尝试次数
    MAX_AUTO_RECOVERY_ATTEMPTS = 3

    # ==================== 降落参数 ====================
    # 最大降落尝试次数
    LANDING_MAX_ATTEMPTS = 2
    # 降落检查间隔，单位：秒
    LANDING_CHECK_INTERVAL = 0.5
    # 降落最大等待时间，单位：秒
    LANDING_MAX_WAIT = 3

    # ==================== 起飞参数 ====================
    # 起飞超时时间，单位：秒
    TAKEOFF_TIMEOUT = 10


# ==================== 相机参数 ====================
# RGB 相机名称（在 AirSim 设置中定义的相机名称）
RGB_CAMERA_NAME = "0"
# 相机类型（0 = 视角相机，1 = 深度相机，2 = 分割相机）
CAMERA_TYPE = 0
# 默认保存图片数量计数器
DEFAULT_IMAGE_COUNT = 0


# ==================== 地面物体名称列表 ====================
# 这些关键词用于碰撞检测时识别地面物体，接触这些物体不会被认为是严重碰撞
GROUND_OBJECTS = ["Road", "Ground", "Terrain", "Grass", "Floor"]


# ==================== 键盘控制参数 ====================
# 键盘控制模式下的移动速度，单位：米/秒
KEYBOARD_VELOCITY = 3
# 键盘控制模式下的旋转速度，单位：度/秒
KEYBOARD_YAW_RATE = 45
# 速度增量，每次按键移动的距离（米）
KEYBOARD_STEP = 3
