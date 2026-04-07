# -*- coding: utf-8 -*-
"""
项目全局配置参数
"""
import os

# ================= CARLA 模拟器配置 =================
CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
CARLA_TIMEOUT = 10.0

# 车辆配置
VEHICLE_MODEL = 'vehicle.tesla.model3'  # 也就是你可以改成 'vehicle.audi.tt' 换个车开
SPAWN_POINT_INDEX = None  # None表示随机，也可以指定固定点

# ================= 传感器配置 =================
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600
CAMERA_FOV = 90
CAMERA_POS_X = 1.5
CAMERA_POS_Z = 2.4

# ================= YOLOv3 模型路径 =================
# 自动获取当前 config.py 的上级目录作为基准
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

YOLO_CONFIG_PATH = os.path.join(BASE_DIR, 'models', 'yolov3.cfg')
YOLO_WEIGHTS_PATH = os.path.join(BASE_DIR, 'models', 'yolov3.weights')
YOLO_NAMES_PATH = os.path.join(BASE_DIR, 'models', 'coco.names')

# ================= 检测与规划参数 =================
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# 规划器参数 (AEB)
SAFE_ZONE_RATIO = 0.4      # 驾驶走廊宽度比例
COLLISION_AREA_THRES = 0.10 # 碰撞预警面积阈值

# ================= 日志配置 =================
LOG_DIR = os.path.join(BASE_DIR, 'runs', 'experiment_1')