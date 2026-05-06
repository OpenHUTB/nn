"""
CARLA恶劣天气鲁棒性测试与自适应感知系统 - 配置文件
"""

# CARLA 服务器
CARLA_HOST = "localhost"
CARLA_PORT = 2000
CARLA_TIMEOUT = 10.0
CARLA_MAP = "Town03"

# 传感器配置
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600
CAMERA_FOV = 90
LIDAR_CHANNELS = 64
LIDAR_RANGE = 50.0

# ===== 恶劣天气测试参数 =====
WEATHER_PROFILES = {
    "clear": {
        "cloudiness": 10.0, "precipitation": 0.0,
        "precipitation_deposits": 0.0, "wind_intensity": 10.0,
        "fog_density": 0.0, "fog_distance": 0.0,
        "wetness": 0.0, "sun_altitude_angle": 70.0,
    },
    "cloudy": {
        "cloudiness": 80.0, "precipitation": 0.0,
        "precipitation_deposits": 0.0, "wind_intensity": 30.0,
        "fog_density": 10.0, "fog_distance": 15.0,
        "wetness": 0.0, "sun_altitude_angle": 40.0,
    },
    "light_rain": {
        "cloudiness": 70.0, "precipitation": 30.0,
        "precipitation_deposits": 30.0, "wind_intensity": 30.0,
        "fog_density": 15.0, "fog_distance": 12.0,
        "wetness": 30.0, "sun_altitude_angle": 30.0,
    },
    "heavy_rain": {
        "cloudiness": 100.0, "precipitation": 100.0,
        "precipitation_deposits": 90.0, "wind_intensity": 80.0,
        "fog_density": 40.0, "fog_distance": 5.0,
        "wetness": 100.0, "sun_altitude_angle": 10.0,
    },
    "fog": {
        "cloudiness": 40.0, "precipitation": 0.0,
        "precipitation_deposits": 0.0, "wind_intensity": 10.0,
        "fog_density": 100.0, "fog_distance": 2.0,
        "wetness": 0.0, "sun_altitude_angle": 60.0,
    },
    "night": {
        "cloudiness": 10.0, "precipitation": 0.0,
        "precipitation_deposits": 0.0, "wind_intensity": 10.0,
        "fog_density": 0.0, "fog_distance": 0.0,
        "wetness": 0.0, "sun_altitude_angle": -90.0,
    },
    "night_rain": {
        "cloudiness": 100.0, "precipitation": 80.0,
        "precipitation_deposits": 80.0, "wind_intensity": 60.0,
        "fog_density": 50.0, "fog_distance": 3.0,
        "wetness": 80.0, "sun_altitude_angle": -90.0,
    },
}

STEPS_PER_WEATHER = 200

# 自适应感知参数
VISIBILITY_THRESHOLD_LOW = 0.3
VISIBILITY_THRESHOLD_HIGH = 0.6
LIDAR_CLUSTER_DISTANCE = 2.0
LIDAR_MIN_CLUSTER_POINTS = 10

# 图像质量评估参数
IMAGE_LAPLACIAN_THRESHOLD = 50.0
IMAGE_BRIGHTNESS_LOW = 40.0
IMAGE_BRIGHTNESS_HIGH = 220.0

# 融合控制参数
CAMERA_WEIGHT_CLEAR = 0.8
LIDAR_WEIGHT_CLEAR = 0.2
CAMERA_WEIGHT_ADVERSE = 0.3
LIDAR_WEIGHT_ADVERSE = 0.7

# 鲁棒性评估
SAFE_DISTANCE = 10.0
COLLISION_DISTANCE = 3.0

# PID 控制
PID_KP = 1.0
PID_KI = 0.01
PID_KD = 0.1
MAX_SPEED = 25.0

# 日志
LOG_LEVEL = "INFO"
