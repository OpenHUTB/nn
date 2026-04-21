"""
配置文件 - 存储所有配置参数
"""
import os
# ==================== 轨迹配置 ====================
TRAJECTORIES = {
    "custom_trajectory": {
        "start": [108.91605377197266,101.3561782836914,0.6136533617973328, 0], 
        "end": [74.64781951904297,-56.014076232910156,0.40768954157829285],
        "description": "自定义轨迹 - 城镇道路"
    },
    "test_trajectory": {
        "start": [98.25761413574219,99.15800476074219,0.4944169223308563, 0],
        "end": [74.64781951904297,-56.014076232910156,0.40768954157829285],
        "description": "测试轨迹 - 直线道路"
    }
}

# 当前使用的轨迹
CURRENT_TRAJECTORY = "custom_trajectory"

def get_current_trajectory():
    """获取当前轨迹配置"""
    if CURRENT_TRAJECTORY in TRAJECTORIES:
        return TRAJECTORIES[CURRENT_TRAJECTORY]
    else:
        print(f"❌ 轨迹 '{CURRENT_TRAJECTORY}' 不存在")
        return None


# ==================== 模型配置 ====================
# 设置基础路径
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)  # 上一级目录
MODELS_DIR = os.path.join(base_dir, 'models')
braking = os.path.join(MODELS_DIR, 'Braking___282.model')
driving = os.path.join(MODELS_DIR, 'Driving__6030.model')
MODEL_PATHS = {
    'braking': braking ,
    'driving': driving
}

# ==================== 动作配置 ====================
ACTION_NAMES = ["刹车", "直行", "左转", "右转", "微左", "微右"]

# ==================== 控制配置 ====================
CONTROL_CONFIG = {
    'max_throttle': 1.0,           # 最大油门
    'max_steer': 1.0,              # 最大转向
    'max_brake': 1.0,              # 最大刹车
    'throttle_increment': 0.2,     # 增加油门增量
    'steer_increment': 0.1,        # 转向增量
    'idle_timeout': 1.0,           # 空闲超时（秒）
}

# ==================== 训练配置 ====================
TOTAL_EPISODES = 3                    # 总共运行的episode数
MAX_STEPS_PER_EPISODE = 2000          # 每个episode最大步数
EPISODE_INTERVAL = 2.0                # episode之间的间隔秒数
UPDATE_RATE = 0.1                     # 降低更新率，让控制更频繁
# ==================== 交通配置 ====================
ENABLE_TRAFFIC = False               # 是否启用交通流（暂时关闭，避免干扰）
TRAFFIC_VEHICLES = 30                 # 交通车辆数量
TRAFFIC_WALKERS = 50                  # 交通行人数量
TRAFFIC_SAFE_MODE = True              # 交通安全模式
TRAFFIC_HYBRID_MODE = True            # 混合物理模式
TRAFFIC_SYNC_MODE = False             # 交通同步模式
TRAFFIC_RESPAWN = False               # 是否重生休眠车辆

# ==================== 可视化配置 ====================
SHOW_ARROW = True                    # 是否显示车辆方向箭头
SHOW_HISTORY_PATH = True             # 是否显示历史行驶路线
SHOW_PLANNED_ROUTE = True            # 是否显示规划路线
SHOW_VEHICLE_MARKER = False           # 是否显示车辆标记

# 箭头设置
ARROW_LENGTH = 5.0                   # 箭头长度（米）
ARROW_THICKNESS = 0.2                # 箭头粗细
ARROW_COLOR = (0, 255, 0)           # 箭头颜色（绿色）
ARROW_BRIGHTNESS = 1.0              # 箭头亮度 (0.0-1.0)

# 历史路线设置
HISTORY_PATH_COLOR = (0, 100, 255)  # 历史路线颜色（蓝色）
HISTORY_PATH_THICKNESS = 0.1         # 历史路线粗细
HISTORY_PATH_MAX_POINTS = 500        # 历史路线最大点数
HISTORY_PATH_FADE_OUT = True         # 是否渐隐效果
HISTORY_PATH_BRIGHTNESS = 0.8        # 历史路线亮度 (0.0-1.0)

# 规划路线设置
PLANNED_ROUTE_COLOR = (255, 0, 0)   # 规划路线颜色（红色）
PLANNED_ROUTE_THICKNESS = 0.15       # 规划路线粗细
PLANNED_ROUTE_BRIGHTNESS = 1.0       # 规划路线亮度 (0.0-1.0)

# 车辆标记设置
VEHICLE_MARKER_COLOR = (0, 255, 0)  # 车辆标记颜色（绿色）
VEHICLE_MARKER_SIZE = 0.4            # 车辆标记大小
VEHICLE_MARKER_BRIGHTNESS = 1.0      # 车辆标记亮度 (0.0-1.0)

# 显示高度（保持不变）
ROUTE_HEIGHT = 0.3                   # 路线显示高度
PATH_HEIGHT = 0.2                    # 路径显示高度
VEHICLE_HEIGHT = 0.25                # 车辆显示高度
# ==================== 视角配置 ====================
TOP_DOWN_HEIGHT = 5.0                # 俯视视角高度（提高）
TOP_DOWN_PITCH = -30.0               # 俯视角 (几乎垂直向下，-90是完全垂直)
SMOOTH_FOLLOW_FACTOR = 0.01          # 视角平滑系数 (0-1，越小越平滑)
MIN_SMOOTH_FACTOR = 0.001            # 最小平滑系数
MAX_SMOOTH_FACTOR = 0.3              # 最大平滑系数
SMOOTH_FACTOR_ADAPTIVE = True        # 是否自适应平滑系数
DISTANCE_THRESHOLD = 1.0             # 距离阈值，超过此值加大平滑系数
FOLLOW_DISTANCE= 8.0                 # 跟随距离（米）
FOLLOW_HEIGHT = 3.0                  # 跟随高度（米）
FOLLOW_PITCH = -20.0                 # 俯视角（度）
VIEW_UPDATE_ENABLED = True           # 是否启用视角更新
VIEW_UPDATE_FPS = 60                 # 视角更新频率 (Hz)
VIEW_UPDATE_THREADED = True          # 是否使用独立线程更新视角
# ==================== 性能配置 ====================
DEBUG_MODE = True                    # 调试模式
FPS_LIMIT = 60                        # FPS限制 (0为无限制，提高性能)
UPDATE_RATE = 0.5                    # 更新率 (1.0 = 每步更新)
USE_SMOOTH_INTERPOLATION = True      # 使用平滑插值
INTERPOLATION_STEPS = 5              # 插值步数
MAX_FRAME_SKIP = 1                   # 最大跳帧数

# ==================== CARLA配置 ====================
CARLA_HOST = "localhost"
CARLA_PORT = 2000
CARLA_TIMEOUT = 20.0

# ==================== 仿真配置 ====================
FIXED_DELTA_SECONDS = 0.0166          # 固定时间步长 (~60 FPS)
SYNCHRONOUS_MODE = False            # 同步模式（影响性能，暂不启用）
NO_RENDERING_MODE = False            # 无渲染模式

# ==================== 传感器配置 ====================
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FOV = 40

def print_config():
    """打印当前配置"""
    print("\n" + "="*60)
    print("当前配置")
    print("="*60)
    
    trajectory = get_current_trajectory()
    if trajectory:
        print(f"轨迹: {trajectory['description']}")
        print(f"起点: {trajectory['start']}")
        print(f"终点: {trajectory['end']}")
    
    print(f"\n模型配置:")
    print(f"  刹车模型: {MODEL_PATHS['braking']}")
    print(f"  驾驶模型: {MODEL_PATHS['driving']}")
    
    print(f"\n训练配置:")
    print(f"  Episodes: {TOTAL_EPISODES}")
    print(f"  最大步数/Episode: {MAX_STEPS_PER_EPISODE}")
    print(f"  Episode间隔: {EPISODE_INTERVAL}s")
    
    print(f"\n视角配置:")
    print(f"  俯视高度: {TOP_DOWN_HEIGHT}m")
    print(f"  俯视角: {TOP_DOWN_PITCH}°")
    print(f"  平滑系数: {SMOOTH_FOLLOW_FACTOR}")
    print(f"  自适应平滑: {'开启' if SMOOTH_FACTOR_ADAPTIVE else '关闭'}")
    print(f"  插值步数: {INTERPOLATION_STEPS}")
    
    print(f"\n性能配置:")
    print(f"  调试模式: {'开启' if DEBUG_MODE else '关闭'}")
    print(f"  FPS限制: {FPS_LIMIT if FPS_LIMIT > 0 else '无限制'}")
    print(f"  固定时间步长: {FIXED_DELTA_SECONDS}s")
    print(f"  同步模式: {'开启' if SYNCHRONOUS_MODE else '关闭'}")
    
    print("="*60)