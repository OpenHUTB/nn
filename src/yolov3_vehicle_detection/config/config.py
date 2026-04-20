class Config:
    # YOLO模型配置
    yolo_cfg_path = "src/yolov3_vehicle_detection/models/yolov3-tiny.cfg"
    yolo_weights_path = "src/yolov3_vehicle_detection/models/yolov3-tiny.weights"
    yolo_names_path = "src/yolov3_vehicle_detection/models/coco.names"
    conf_thres = 0.5
    nms_thres = 0.4

    # CARLA配置
    carla_host = "127.0.0.1"
    carla_port = 2000
    carla_timeout = 30.0
    CARLA_HOST = carla_host
    CARLA_PORT = carla_port
    CARLA_TIMEOUT = carla_timeout
    VEHICLE_MODEL = "vehicle.tesla.model3"

    # 摄像头配置
    camera_width = 800
    camera_height = 600
    camera_fov = 110
    CAMERA_WIDTH = camera_width
    CAMERA_HEIGHT = camera_height
    CAMERA_FOV = camera_fov
    CAMERA_POS_X = 0.3
    CAMERA_POS_Z = 1.3

    # 安全区域配置
    SAFE_ZONE_RATIO = 0.4
    COLLISION_AREA_THRES = 0.05

    # 日志配置
    log_dir = "logs"
    LOG_DIR = log_dir


# 创建Config实例供模块导入使用
config = Config()