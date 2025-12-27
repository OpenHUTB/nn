"""
AirSimNH 无人机项目配置文件
所有可调参数集中在此管理

配置结构说明：
1. 物体检测配置：PERCEPTION['{COLOR}_OBJECT_DETECTION']
   - 使用 get_object_detection_config(color_type) 函数获取
   - 或直接访问 PERCEPTION['RED_OBJECT_DETECTION'] 等

2. 物体探索配置：INTELLIGENT_DECISION['{COLOR}_OBJECT_EXPLORATION']
   - 使用 get_object_exploration_config(color_type) 函数获取
   - 或直接访问 INTELLIGENT_DECISION['RED_OBJECT_EXPLORATION'] 等

3. 颜色范围配置：CAMERA['{COLOR}_COLOR_RANGE']
   - 使用 get_color_range_config(color_type) 函数获取
   - 或直接访问 CAMERA['RED_COLOR_RANGE'] 等

扩展新颜色类型：
  1. 在 PERCEPTION 中添加 '{COLOR}_OBJECT_DETECTION' 配置
  2. 在 INTELLIGENT_DECISION 中添加 '{COLOR}_OBJECT_EXPLORATION' 配置
  3. 在 CAMERA 中添加 '{COLOR}_COLOR_RANGE' 配置
  4. 使用提供的辅助函数 create_object_detection_config() 和 create_object_exploration_config()
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
    },

    # ========== 物体检测配置说明 ==========
    # 所有颜色类型的物体检测配置使用统一结构，包含以下字段：
    #   - ENABLED: 是否启用检测 (bool)
    #   - MIN_AREA: 最小检测面积（像素）(int)
    #   - MAX_AREA: 最大检测面积（像素）(int)
    #   - UPDATE_INTERVAL: 检测更新间隔（秒）(float)
    #   - MEMORY_TIME: 物体记忆时间（秒）(float)
    #
    # 扩展新颜色类型时，可以参考以下模板或使用 create_object_detection_config() 函数
    '_OBJECT_DETECTION_TEMPLATE': {
        'ENABLED': True,  # 启用物体检测
        'MIN_AREA': 50,  # 最小检测面积（像素）
        'MAX_AREA': 10000,  # 最大检测面积（像素）
        'UPDATE_INTERVAL': 1.0,  # 检测更新间隔（秒）
        'MEMORY_TIME': 5.0,  # 物体记忆时间（秒），避免重复计数
    },
    
    # 红色物体检测参数
    'RED_OBJECT_DETECTION': {
        'ENABLED': True,  # 启用红色物体检测
        'MIN_AREA': 50,  # 最小检测面积（像素）
        'MAX_AREA': 10000,  # 最大检测面积（像素）
        'UPDATE_INTERVAL': 1.0,  # 检测更新间隔（秒）
        'MEMORY_TIME': 5.0,  # 物体记忆时间（秒），避免重复计数
    },

    # 蓝色物体检测参数
    'BLUE_OBJECT_DETECTION': {
        'ENABLED': True,  # 启用蓝色物体检测
        'MIN_AREA': 50,  # 最小检测面积（像素）
        'MAX_AREA': 10000,  # 最大检测面积（像素）
        'UPDATE_INTERVAL': 1.0,  # 检测更新间隔（秒）
        'MEMORY_TIME': 5.0,  # 物体记忆时间（秒）
    },

    # 黑色物体检测参数
    'BLACK_OBJECT_DETECTION': {
        'ENABLED': True,  # 启用黑色物体检测
        'MIN_AREA': 50,  # 最小检测面积（像素）
        'MAX_AREA': 10000,  # 最大检测面积（像素）
        'UPDATE_INTERVAL': 1.0,  # 检测更新间隔（秒）
        'MEMORY_TIME': 5.0,  # 物体记忆时间（秒）
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

    # ========== 物体探索配置说明 ==========
    # 所有颜色类型的物体探索配置使用统一结构，包含以下字段：
    #   - ATTRACTION_GAIN: 物体吸引力增益 (float)
    #   - DETECTION_RADIUS: 检测半径（米）(float)
    #   - MIN_DISTANCE: 最小接近距离（米）(float)
    #   - EXPLORATION_BONUS: 探索奖励分数 (float)
    #
    # 扩展新颜色类型时，可以参考以下模板或使用 create_object_exploration_config() 函数
    '_OBJECT_EXPLORATION_TEMPLATE': {
        'ATTRACTION_GAIN': 1.0,  # 物体吸引力增益
        'DETECTION_RADIUS': 8.0,  # 检测半径（米）
        'MIN_DISTANCE': 2.0,  # 最小接近距离（米）
        'EXPLORATION_BONUS': 0.2,  # 探索奖励分数
    },
    
    # 红色物体探索参数（优先级最高）
    'RED_OBJECT_EXPLORATION': {
        'ATTRACTION_GAIN': 1.5,  # 红色物体吸引力增益
        'DETECTION_RADIUS': 10.0,  # 检测半径（米）
        'MIN_DISTANCE': 2.0,  # 最小接近距离（米）
        'EXPLORATION_BONUS': 0.5,  # 探索奖励分数
    },

    # 蓝色物体探索参数
    'BLUE_OBJECT_EXPLORATION': {
        'ATTRACTION_GAIN': 1.2,  # 蓝色物体吸引力增益
        'DETECTION_RADIUS': 8.0,  # 检测半径（米）
        'MIN_DISTANCE': 2.0,  # 最小接近距离（米）
        'EXPLORATION_BONUS': 0.3,  # 探索奖励分数
    },

    # 黑色物体探索参数
    'BLACK_OBJECT_EXPLORATION': {
        'ATTRACTION_GAIN': 1.0,  # 黑色物体吸引力增益
        'DETECTION_RADIUS': 8.0,  # 检测半径（米）
        'MIN_DISTANCE': 2.0,  # 最小接近距离（米）
        'EXPLORATION_BONUS': 0.2,  # 探索奖励分数
    }
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

# ==================== 窗口显示参数 ====================
DISPLAY = {
    # 前视窗口参数
    'FRONT_VIEW_WINDOW': {
        'NAME': "无人机前视画面",
        'WIDTH': 640,
        'HEIGHT': 480,
        'ENABLE_SHARPENING': True,  # 启用图像锐化，改善模糊
        'SHOW_INFO_OVERLAY': True,  # 显示信息叠加层
        'REFRESH_RATE_MS': 30,  # 刷新率 (毫秒)，建议30-50
        'SHOW_RED_OBJECTS': True,  # 在画面中标记红色物体
        'SHOW_BLUE_OBJECTS': True,  # 在画面中标记蓝色物体
        'SHOW_BLACK_OBJECTS': True,  # 在画面中标记黑色物体（新增）
        # 内存优化参数
        'QUEUE_MAXSIZE': 2,  # 图像队列最大大小，减少内存占用（原为3）
        'REDUCE_IMAGE_COPY': True,  # 减少图像复制，仅在必要时复制
    },

    # 信息显示窗口参数
    'INFO_WINDOW': {
        'NAME': "无人机信息面板",
        'WIDTH': 800,
        'HEIGHT': 600,
        'BACKGROUND_COLOR': (20, 20, 30),  # 深蓝灰色背景
        'TEXT_COLOR': (220, 220, 255),  # 浅蓝色文字
        'HIGHLIGHT_COLOR': (0, 200, 255),  # 青色高亮
        'WARNING_COLOR': (0, 100, 255),  # 橙色警告
        'SUCCESS_COLOR': (0, 255, 150),  # 绿色成功
        'REFRESH_RATE_MS': 100,  # 信息窗口刷新率
        'SHOW_GRID': True,  # 显示探索网格
        'GRID_SIZE': 300,  # 网格显示大小
        'SHOW_OBJECTS_STATS': True,  # 显示物体统计
        'SHOW_SYSTEM_STATS': True,  # 显示系统统计
        'SHOW_PERFORMANCE': True,  # 显示性能信息
        'SHOW_TRAJECTORY': True,  # 显示运动轨迹图
        'TRAJECTORY_SIZE': 280,  # 轨迹图显示大小（像素）
        'TRAJECTORY_MAX_POINTS': 1000,  # 轨迹最大记录点数
        'TRAJECTORY_LINE_COLOR': (0, 255, 255),  # 轨迹线颜色（青色）
        'TRAJECTORY_CURRENT_COLOR': (0, 255, 0),  # 当前位置颜色（绿色）
        'TRAJECTORY_START_COLOR': (255, 255, 0),  # 起始位置颜色（黄色）
    }
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
CAMERA = {
    'DEFAULT_NAME': "0",  # 默认相机名称
    'POSSIBLE_NAMES': ["0", "1", "front_center", "front", "CameraActor_0"],

    # 深度相机参数
    'DEPTH_FOV_DEG': 90,  # 深度相机视野 (度)
    'MAX_DEPTH_RANGE': 100.0,  # 最大深度范围 (米),

    # 红色物体检测颜色范围（HSV空间）
    'RED_COLOR_RANGE': {
        'LOWER1': [0, 120, 70],    # 红色下限1（0-10度）
        'UPPER1': [10, 255, 255],  # 红色上限1
        'LOWER2': [170, 120, 70],  # 红色下限2（170-180度）
        'UPPER2': [180, 255, 255], # 红色上限2
    },

    # 蓝色物体检测颜色范围（HSV空间）（新增）
    'BLUE_COLOR_RANGE': {
        'LOWER': [100, 150, 50],    # 蓝色下限
        'UPPER': [130, 255, 255],   # 蓝色上限
    },

    # 黑色物体检测颜色范围（HSV空间）（新增）
    'BLACK_COLOR_RANGE': {
        'LOWER': [0, 0, 0],         # 黑色下限（色相任意，饱和度和亮度都很低）
        'UPPER': [180, 255, 50],    # 黑色上限（亮度阈值50，避免过暗）
    }
}

# ==================== 调试参数 ====================
DEBUG = {
    'SAVE_PERCEPTION_IMAGES': False,  # 是否保存感知图像用于调试
    'IMAGE_SAVE_INTERVAL': 50,  # 图像保存间隔 (循环次数)
    'LOG_DECISION_DETAILS': True,  # 是否记录详细决策信息
    'SAVE_GRID_VISUALIZATION': True,  # 是否保存网格可视化
    'LOG_VECTOR_FIELD': False,  # 是否记录向量场详细信息
    'PERFORMANCE_PROFILING': False,  # 是否启用性能分析
    'SAVE_RED_OBJECT_IMAGES': False,  # 是否保存检测到红色物体的图像
    'SAVE_BLUE_OBJECT_IMAGES': False,  # 是否保存检测到蓝色物体的图像（新增）
    'SAVE_BLACK_OBJECT_IMAGES': False,  # 是否保存检测到黑色物体的图像（新增）
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
    'RECORD_RED_OBJECTS': True,           # 记录红色物体信息
    'RECORD_BLUE_OBJECTS': True,          # 记录蓝色物体信息（新增）
    'RECORD_BLACK_OBJECTS': True,         # 记录黑色物体信息（新增）
    # 内存优化参数
    'MAX_FLIGHT_DATA_BUFFER': 500,        # 最大飞行数据缓冲区大小（条），超过后自动保存并清空
    'MAX_OBJECTS_BUFFER': 200,            # 最大物体记录缓冲区大小（个）
    'MAX_EVENTS_BUFFER': 100,             # 最大事件记录缓冲区大小（个）
    'AUTO_SAVE_INTERVAL': 60.0,           # 自动保存间隔（秒），定期保存数据到文件
}

# ==================== 性能监控参数 ====================
PERFORMANCE = {
    'ENABLE_REALTIME_METRICS': True,      # 启用实时性能监控
    'CPU_WARNING_THRESHOLD': 80.0,        # CPU使用率警告阈值（%）
    'MEMORY_WARNING_THRESHOLD': 80.0,     # 内存使用率警告阈值（%）
    'LOOP_TIME_WARNING_THRESHOLD': 0.2,   # 循环时间警告阈值（秒）
    'SAVE_PERFORMANCE_REPORT': True,      # 保存性能报告
    'REPORT_INTERVAL': 30.0,              # 性能报告间隔（秒）
    # 内存优化参数
    'MAX_METRICS_BUFFER': 500,            # 最大性能指标缓冲区大小（个），减少内存占用
}

# ==================== 配置辅助函数 ====================
def get_object_detection_config(color_type: str) -> dict:
    """
    获取指定颜色类型的物体检测配置
    
    Args:
        color_type: 颜色类型 ('red', 'blue', 'black')
    
    Returns:
        dict: 物体检测配置字典
    
    Example:
        config = get_object_detection_config('red')
        # 返回 PERCEPTION['RED_OBJECT_DETECTION']
    """
    config_key = f'{color_type.upper()}_OBJECT_DETECTION'
    if config_key not in PERCEPTION:
        raise ValueError(f"不支持的颜色类型: {color_type}，支持的类型: ['red', 'blue', 'black']")
    return PERCEPTION[config_key]


def get_object_exploration_config(color_type: str) -> dict:
    """
    获取指定颜色类型的物体探索配置
    
    Args:
        color_type: 颜色类型 ('red', 'blue', 'black')
    
    Returns:
        dict: 物体探索配置字典
    
    Example:
        config = get_object_exploration_config('blue')
        # 返回 INTELLIGENT_DECISION['BLUE_OBJECT_EXPLORATION']
    """
    config_key = f'{color_type.upper()}_OBJECT_EXPLORATION'
    if config_key not in INTELLIGENT_DECISION:
        raise ValueError(f"不支持的颜色类型: {color_type}，支持的类型: ['red', 'blue', 'black']")
    return INTELLIGENT_DECISION[config_key]


def get_color_range_config(color_type: str) -> dict:
    """
    获取指定颜色类型的颜色范围配置
    
    Args:
        color_type: 颜色类型 ('red', 'blue', 'black')
    
    Returns:
        dict: 颜色范围配置字典
    
    Example:
        config = get_color_range_config('red')
        # 返回 CAMERA['RED_COLOR_RANGE']
    """
    config_key = f'{color_type.upper()}_COLOR_RANGE'
    if config_key not in CAMERA:
        raise ValueError(f"不支持的颜色类型: {color_type}，支持的类型: ['red', 'blue', 'black']")
    return CAMERA[config_key]


def get_all_color_types() -> list:
    """
    获取所有支持的颜色类型列表
    
    Returns:
        list: 支持的颜色类型列表 ['red', 'blue', 'black']
    """
    return ['red', 'blue', 'black']


def create_object_detection_config(enabled: bool = True, min_area: int = 50, 
                                   max_area: int = 10000, update_interval: float = 1.0,
                                   memory_time: float = 5.0) -> dict:
    """
    创建新的物体检测配置（用于扩展新颜色类型）
    
    Args:
        enabled: 是否启用检测
        min_area: 最小检测面积（像素）
        max_area: 最大检测面积（像素）
        update_interval: 检测更新间隔（秒）
        memory_time: 物体记忆时间（秒）
    
    Returns:
        dict: 新的物体检测配置字典
    
    Example:
        # 添加黄色物体检测配置
        PERCEPTION['YELLOW_OBJECT_DETECTION'] = create_object_detection_config(
            enabled=True,
            min_area=50,
            max_area=10000,
            update_interval=1.0,
            memory_time=5.0
        )
    """
    return {
        'ENABLED': enabled,
        'MIN_AREA': min_area,
        'MAX_AREA': max_area,
        'UPDATE_INTERVAL': update_interval,
        'MEMORY_TIME': memory_time,
    }


def create_object_exploration_config(attraction_gain: float = 1.0,
                                     detection_radius: float = 8.0,
                                     min_distance: float = 2.0,
                                     exploration_bonus: float = 0.2) -> dict:
    """
    创建新的物体探索配置（用于扩展新颜色类型）
    
    Args:
        attraction_gain: 吸引力增益
        detection_radius: 检测半径（米）
        min_distance: 最小接近距离（米）
        exploration_bonus: 探索奖励分数
    
    Returns:
        dict: 新的物体探索配置字典
    
    Example:
        # 添加黄色物体探索配置
        INTELLIGENT_DECISION['YELLOW_OBJECT_EXPLORATION'] = create_object_exploration_config(
            attraction_gain=1.3,
            detection_radius=9.0,
            min_distance=2.0,
            exploration_bonus=0.4
        )
    """
    return {
        'ATTRACTION_GAIN': attraction_gain,
        'DETECTION_RADIUS': detection_radius,
        'MIN_DISTANCE': min_distance,
        'EXPLORATION_BONUS': exploration_bonus,
    }