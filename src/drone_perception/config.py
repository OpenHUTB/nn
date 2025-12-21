# config.py
"""
AirSimNH 无人机项目配置文件
所有可调参数集中在此管理
"""

import math

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

# ==================== 智能决策参数 ====================
INTELLIGENT_DECISION = {
    # 向量场算法参数
    'VECTOR_FIELD_RADIUS': 8.0,  # 向量场影响半径 (米)
    'OBSTACLE_REPULSION_GAIN': 3.0,  # 障碍物排斥增益
    'GOAL_ATTRACTION_GAIN': 2.0,  # 目标吸引力增益
    'SMOOTHING_FACTOR': 0.3,  # 向量平滑因子
    'MIN_TURN_ANGLE_DEG': 10,  # 最小转弯角度 (度)
    'MAX_TURN_ANGLE_DEG': 60,  # 最大转弯角度 (度)

    # 探索网格参数
    'GRID_RESOLUTION': 2.0,  # 网格分辨率 (米)
    'GRID_SIZE': 50,  # 网格大小 (单元格数)
    'INFORMATION_GAIN_DECAY': 0.95,  # 信息增益衰减率
    'EXPLORATION_FRONTIER_THRESHOLD': 0.3,  # 探索前沿阈值

    # 控制参数
    'PID_KP': 1.5,  # 比例系数
    'PID_KI': 0.05,  # 积分系数
    'PID_KD': 0.2,  # 微分系数
    'SMOOTHING_WINDOW_SIZE': 5,  # 平滑窗口大小

    # 自适应参数
    'ADAPTIVE_SPEED_ENABLED': True,  # 启用自适应速度
    'MIN_SPEED_FACTOR': 0.3,  # 最小速度因子
    'MAX_SPEED_FACTOR': 1.5,  # 最大速度因子

    # 探索策略权重
    'MEMORY_WEIGHT': 0.7,  # 记忆权重 (避免重复访问)
    'CURIOUSITY_WEIGHT': 0.3,  # 好奇心权重 (探索新区域)

    # 目标管理
    'TARGET_LIFETIME': 15.0,  # 目标有效期 (秒)
    'TARGET_REACHED_DISTANCE': 3.0,  # 目标到达判定距离 (米)
}

# ==================== 手动控制参数 ====================
MANUAL = {
    'CONTROL_SPEED': 3.0,  # 水平移动速度 (米/秒)
    'ALTITUDE_SPEED': 2.0,  # 垂直移动速度 (米/秒)
    'YAW_SPEED': 45.0,  # 偏航角速度 (度/秒)
    'ENABLE_AUTO_HOVER': True,  # 松开按键时自动悬停
    'DISPLAY_CONTROLS': True,  # 在画面显示控制说明
    'SAFETY_ENABLED': True,  # 启用安全限制 (高度、速度限制)
    'MAX_MANUAL_SPEED': 5.0,  # 最大手动控制速度
    'MIN_ALTITUDE_LIMIT': -5.0,  # 最低飞行高度限制
    'MAX_ALTITUDE_LIMIT': -30.0,  # 最高飞行高度限制
}

# ==================== 前视窗口参数 ====================
DISPLAY = {
    'WINDOW_WIDTH': 640,  # 窗口宽度 (像素)
    'WINDOW_HEIGHT': 480,  # 窗口高度 (像素)
    'ENABLE_SHARPENING': True,  # 启用图像锐化，改善模糊
    'SHOW_INFO_OVERLAY': True,  # 显示信息叠加层
    'REFRESH_RATE_MS': 30,  # 刷新率 (毫秒)，建议30-50
    'GRID_DISPLAY_SIZE': 150,  # 探索网格显示大小 (像素)
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

    'EMERGENCY_RESPONSE_TIME': 10.0,  # 紧急响应超时时间 (秒)
}

# ==================== 相机配置 ====================
# 注意：这里的相机名称需要与AirSim环境中的相机名称匹配
CAMERA = {
    'DEFAULT_NAME': "0",  # 默认相机名称
    # 如果"0"无效，可以尝试以下名称:
    'POSSIBLE_NAMES': ["0", "1", "front_center", "front", "CameraActor_0"],

    # 深度相机参数
    'DEPTH_FOV_DEG': 90,  # 深度相机视野 (度)
    'MAX_DEPTH_RANGE': 100.0,  # 最大深度范围 (米)
}

# ==================== 调试参数 ====================
DEBUG = {
    'SAVE_PERCEPTION_IMAGES': False,  # 是否保存感知图像用于调试
    'IMAGE_SAVE_INTERVAL': 50,  # 图像保存间隔 (循环次数)
    'LOG_DECISION_DETAILS': True,  # 是否记录详细决策信息
    'SAVE_GRID_VISUALIZATION': True,  # 是否保存网格可视化
    'LOG_VECTOR_FIELD': False,  # 是否记录向量场详细信息
    'PERFORMANCE_PROFILING': False,  # 是否启用性能分析
}

# ==================== 数据记录参数 ====================
DATA_RECORDING = {
    'ENABLED': True,                      # 启用数据记录
    'RECORD_INTERVAL': 0.2,               # 记录间隔（秒）
    'SAVE_TO_CSV': True,                  # 保存为CSV格式
    'SAVE_TO_JSON': True,                 # 保存为JSON格式
    'CSV_FILENAME': 'flight_data.csv',    # CSV文件名
    'JSON_FILENAME': 'flight_data.json',  # JSON文件名
    'PERFORMANCE_MONITORING': True,       # 启用性能监控
    'SYSTEM_METRICS_INTERVAL': 5.0,       # 系统指标记录间隔（秒）
}

# ==================== 性能监控参数 ====================
PERFORMANCE = {
    'ENABLE_REALTIME_METRICS': True,      # 启用实时性能监控
    'CPU_WARNING_THRESHOLD': 80.0,        # CPU使用率警告阈值（%）
    'MEMORY_WARNING_THRESHOLD': 80.0,     # 内存使用率警告阈值（%）
    'LOOP_TIME_WARNING_THRESHOLD': 0.2,   # 循环时间警告阈值（秒）
    'SAVE_PERFORMANCE_REPORT': True,      # 保存性能报告
    'REPORT_INTERVAL': 30.0,              # 性能报告间隔（秒）
}