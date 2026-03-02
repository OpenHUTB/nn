# -*- coding: utf-8 -*-
"""
项目全局配置参数
"""

# CARLA 模拟器配置
CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
CARLA_TIMEOUT = 10.0

# 传感器配置
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600
CAMERA_FOV = 90

# YOLOv3 模型相关路径
YOLO_CONFIG_PATH = 'models/yolov3.cfg'
YOLO_WEIGHTS_PATH = 'models/yolov3.weights'
YOLO_NAMES_PATH = 'models/coco.names'

# 检测阈值
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4