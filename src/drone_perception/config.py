# config.py
"""
AirSimNH 无人机项目配置文件
所有可调参数集中在此管理
"""

# ==================== 飞行与探索参数 ====================
EXPLORATION = {
    'TOTAL_TIME': 120,  # 总探索时间 (秒)，建议120-180
    'PREFERRED_SPEED': 2.5,  # 巡航速度 (米/秒)，建议2.0-3.0
    'BASE_HEIGHT': -15.0,  # 基础飞行高度 (米，负值)，建议-12到-18
    'MAX_ALTITUDE': -30.0,  # 最大海拔 (米)，限制无人机最高飞行高度
    'MIN_ALTITUDE': -5.0,  # 最小海拔 (米)，限制无人机最低飞行高度
    'TAKEOFF_HEIGHT': -10.0,  # 起飞目标高度，建议-8到-12
}

# ==================== 感知参数 ====================
PERCEPTION = {
    'DEPTH_NEAR_THRESHOLD': 5.0,  # 近距离警报阈值 (米)，小于此值触发避障
    'DEPTH_SAFE_THRESHOLD': 10.0,  # 安全距离阈值 (米)，大于此值认为方向安全
    'MIN_GROUND_CLEARANCE': 2.0,  # 最小离地间隙 (米)，防止撞地
    'MAX_PITCH_ANGLE_DEG': 15,  # 最大允许俯仰角 (度)

    # 深度图像扫描角度 (度)，用于多方向安全检测
    'SCAN_ANGLES': [-60, -45, -30, -15, 0, 15, 30, 45, 60],

    # 高度推荐策略
    'HEIGHT_STRATEGY': {
        'STEEP_SLOPE': -20.0,  # 陡峭地形高度
        'OPEN_SPACE': -12.0,  # 开阔地带高度
        'DEFAULT': -15.0,  # 默认高度
        'SLOPE_THRESHOLD': 5.0,  # 坡度阈值，大于此值认为地形陡峭
        'OPENNESS_THRESHOLD': 0.7,  # 开阔度阈值，大于此值认为开阔
    }
}

# ==================== 前视窗口参数 ====================
DISPLAY = {
    'WINDOW_WIDTH': 640,  # 窗口宽度
    'WINDOW_HEIGHT': 480,  # 窗口高度
    'ENABLE_SHARPENING': True,  # 启用图像锐化，改善模糊
    'SHOW_INFO_OVERLAY': True,  # 显示信息叠加层
    'REFRESH_RATE_MS': 30,  # 刷新率 (毫秒)，建议30-50
}

# ==================== 系统与安全参数 ====================
SYSTEM = {
    'LOG_LEVEL': 'INFO',  # 日志级别: DEBUG, INFO, WARNING, ERROR
    'LOG_TO_FILE': True,  # 是否保存日志到文件
    'LOG_FILENAME': 'drone_log.txt',  # 日志文件名

    'MAX_RECONNECT_ATTEMPTS': 3,  # 最大重连尝试次数
    'RECONNECT_DELAY': 2.0,  # 重连延迟 (秒)

    'ENABLE_HEALTH_CHECK': True,  # 启用健康检查
    'HEALTH_CHECK_INTERVAL': 20,  # 健康检查间隔 (循环次数)
}

# ==================== 相机配置 ====================
# 注意：这里的相机名称需要与AirSim环境中的相机名称匹配
CAMERA = {
    'DEFAULT_NAME': "0",  # 默认相机名称
    # 如果"0"无效，可以尝试以下名称:
    # 'POSSIBLE_NAMES': ["0", "1", "front_center", "front", "CameraActor_0"]
}

# ==================== 调试参数 ====================
DEBUG = {
    'SAVE_PERCEPTION_IMAGES': False,  # 是否保存感知图像用于调试
    'IMAGE_SAVE_INTERVAL': 50,  # 图像保存间隔 (循环次数)
    'LOG_DECISION_DETAILS': False,  # 是否记录详细决策信息
}