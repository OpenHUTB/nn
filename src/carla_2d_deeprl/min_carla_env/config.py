"""
min-carla-env 独立配置文件
=====================
将 CONFIG 抽离成独立配置模块，支持：
- 动态修改观测尺寸
- 天气预设切换
- 地图选择
- 奖励系数调优
- 渲染开关 / 快速模式 / 调试模式 分离控制
"""

# carla 为可选依赖，仅在需要天气预设时导入
try:
    import carla
    _CARLA_AVAILABLE = True
except ImportError:
    _CARLA_AVAILABLE = False
    carla = None  # type: ignore

# ============================================================
# 观测尺寸配置
# ============================================================
OBS_CONFIG = {
    "width": 480,        # 观测画面宽度
    "height": 480,       # 观测画面高度
    "channels": 1,       # 语义分割为单通道（RGB 为 3）
}

# ============================================================
# 环境基础配置
# ============================================================
ENV_CONFIG = {
    "max_step": 90000,          # 单 episode 最大步数
    "target_speed": 20,         # 目标车速 (km/h)
    "throttle": 0.3,            # 油门大小
    "brake_at_target": 0.2,     # 超速时刹车力度
    "stuck_distance": 0.05,     # 判定卡住的距离阈值 (m)
    "stuck_max_count": 20,      # 卡住多少步后终止
}

# ============================================================
# 奖励系数配置（调优后更稳定的版本）
# ============================================================
REWARD_CONFIG = {
    # 车道中心奖励
    "lane_center_reward": 0.5,        # 在车道中心时的正奖励
    "lane_center_threshold": 0.5,     # 判定"在车道中心"的距离阈值 (m)
    "dist_penalty_scale": 1.0,        # 偏离中心时指数惩罚的缩放系数
    "dist_penalty_clip": 1000.0,      # 距离惩罚的裁剪上限

    # 终止事件惩罚
    "collision_penalty_mult": 2.0,    # 碰撞时奖励乘数
    "lane_violation_penalty_mult": 2.0,  # 压实线时奖励乘数
    "sidewalk_penalty_mult": 2.0,     # 上人行道时奖励乘数
    "stuck_penalty": 100.0,           # 卡住时的固定惩罚

    # 速度奖励（新增，让训练更稳定）
    "speed_reward_scale": 0.01,       # 速度奖励缩放（鼓励维持目标车速）
}

# ============================================================
# 渲染 / 快速模式 / 调试 开关（分离控制）
# ============================================================
RENDER_CONFIG = {
    "render": True,        # 是否渲染画面（False=无渲染模式，大幅加速训练）
    "fast": False,         # 快速仿真模式（fixed_delta_seconds=0.05）
    "debug": False,        # 调试模式（保存 i.png / s.png 到磁盘）
}

# ============================================================
# 天气预设（仅在 carla 可用时填充）
# ============================================================
WEATHER_PRESETS = {}
if _CARLA_AVAILABLE:
    WEATHER_PRESETS = {
        "clear_noon": carla.WeatherParameters.ClearNoon,
        "cloudy_noon": carla.WeatherParameters.CloudyNoon,
        "wet_noon": carla.WeatherParameters.WetNoon,
        "wet_cloudy_noon": carla.WeatherParameters.WetCloudyNoon,
        "mid_rainy_noon": carla.WeatherParameters.MidRainyNoon,
        "hard_rain_noon": carla.WeatherParameters.HardRainNoon,
        "soft_rain_noon": carla.WeatherParameters.SoftRainNoon,
        "clear_sunset": carla.WeatherParameters.ClearSunset,
        "cloudy_sunset": carla.WeatherParameters.CloudySunset,
        "wet_sunset": carla.WeatherParameters.WetSunset,
    }

# ============================================================
# 地图选项
# ============================================================
MAP_OPTIONS = {
    "town01": "Town01",
    "town02": "Town02",
    "town03": "Town03",
    "town04": "Town04",
    "town05": "Town05",
    "town07": "Town07",
}

# ============================================================
# 动作空间
# ============================================================
ACTIONS = {
    0: [0.0,  0.0],    # 直行 (Coast)
    1: [0.0, -0.5],    # 左转 (Turn Left)
    2: [0.0,  0.5],    # 右转 (Turn Right)
}

# ============================================================
# 向后兼容的 CONFIG 字典（整合所有配置）
# ============================================================
CONFIG = {
    **OBS_CONFIG,
    **ENV_CONFIG,
    **RENDER_CONFIG,
}
