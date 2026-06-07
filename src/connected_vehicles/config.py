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
GUI_WINDOW_SIZE = "400x300+20+20"
GUI_UPDATE_INTERVAL_MS = 50  # GUI更新间隔
GUI_TITLE = "车辆实时状态监控"

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