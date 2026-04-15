# config.py
"""无人机飞行配置参数"""

class FlightConfig:
    # 飞行参数
    TAKEOFF_HEIGHT = -3  # 起飞高度（负值表示向上）
    FLIGHT_VELOCITY = 3  # 飞行速度 (m/s)
    MAX_FLIGHT_TIME = 60  # 最大飞行时间（秒）
    
    # 碰撞检测参数
    COLLISION_COOLDOWN = 1.0  # 碰撞冷却时间（秒）
    GROUND_HEIGHT_THRESHOLD = 1.5  # 地面判断阈值（米）
    ARRIVAL_TOLERANCE = 1.0  # 到达目标点的容差（米）
    
    # 降落参数
    LANDING_MAX_ATTEMPTS = 3  # 最大降落尝试次数
    LANDING_CHECK_INTERVAL = 0.5  # 降落检查间隔（秒）
    LANDING_MAX_WAIT = 5  # 降落最大等待时间（秒）
    
    # 起飞参数
    TAKEOFF_TIMEOUT = 10  # 起飞超时时间（秒）


# 地面物体名称关键词（碰撞时忽略）
GROUND_OBJECTS = ["Road", "Ground", "Terrain", "Grass", "Floor"]