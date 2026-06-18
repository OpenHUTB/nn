# 全局配置项，统一管理可配置参数
import carla

# CARLA连接配置
CARLA_HOST = "127.0.0.1"
CARLA_PORT = 2000
CARLA_TIMEOUT = 10.0

# 车辆控制配置
MAX_SPEED_KMH = 100.0  # 最大车速限制
SPAWN_POINT_OFFSET = 10.0  # 车辆生成位置后退距离
STEER_ANGLE = 0.5  # 转向角度
BRAKE_INTENSITY = 1.0  # 刹车强度

# GUI配置
GUI_WINDOW_SIZE = "500x450+20+20"  # 扩大窗口适配新数据
GUI_UPDATE_INTERVAL_MS = 50  # GUI更新间隔
GUI_TITLE = "车辆实时状态监控 - 含驾驶员体征"

# 天气配置
WEATHER_LIST = ["clear", "rain", "fog", "night"]
DEFAULT_WEATHER = "clear"

# 红绿灯检测配置
TRAFFIC_LIGHT_DETECT_DISTANCE = 50.0  # 红绿灯检测最大距离
TRAFFIC_LIGHT_FILTER = "traffic.traffic_light"

# 碰撞检测配置
COLLISION_SENSOR_BP = "sensor.other.collision"
COLLISION_LOG_FILE = "collision_logs.txt"

# 日志配置
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# 驾驶员体征监测配置
VITALS_BASE_HEART_RATE = 75.0  # 基础心率(次/分钟)
VITALS_BASE_BLOOD_PRESSURE = (120, 80)  # 基础血压(收缩压/舒张压)
VITALS_BASE_FATIGUE = 0.0  # 基础疲惫度(0-100)

# 体征影响因子
VITALS_SPEED_FACTOR = {
    "heart_rate": 0.3,    # 车速每增加1km/h，心率增加0.3
    "blood_pressure": 0.2 # 车速每增加1km/h，血压收缩压增加0.2
}
VITALS_TIME_FACTOR = {
    "heart_rate": 0.1,    # 每驾驶1分钟，心率增加0.1
    "fatigue": 0.2        # 每驾驶1分钟，疲惫度增加0.2
}
VITALS_WEATHER_FACTORS = {
    "clear": 1.0,   # 晴天无额外影响
    "rain": 1.2,    # 雨天影响因子
    "fog": 1.3,     # 雾天影响因子
    "night": 1.15   # 夜间影响因子
}
VITALS_COLLISION_FACTOR = {
    "heart_rate": 0.8,    # 碰撞车速每增加1km/h，心率额外增加0.8
    "blood_pressure": 0.5 # 碰撞车速每增加1km/h，血压收缩压额外增加0.5
}