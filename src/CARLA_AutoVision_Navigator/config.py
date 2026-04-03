# -*- coding: utf-8 -*-
"""
Project: CARLA AutoVision Navigator
Module: Configuration Management
Version: v1.0.0
Description: 全局参数配置文件，统一管理服务器连接、传感器参数、模型配置及 PID 控制增益。
Author: wangadsa
License: MIT License
"""

# ==============================================================================
# -- CARLA Server Settings ----------------------------------------------------
# ==============================================================================
CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
TIMEOUT = 10.0

# ==============================================================================
# -- Vehicle & Sensor Settings ------------------------------------------------
# ==============================================================================
VEHICLE_FILTER = 'vehicle.tesla.model3'
SPAWN_POINT_INDEX = 1

CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600
CAMERA_FOV = 90

# ==============================================================================
# -- YOLOv3 Model Settings ----------------------------------------------------
# ==============================================================================
MODEL_CFG = "models/yolov3.cfg"
MODEL_WEIGHTS = "models/yolov3.weights"
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# ==============================================================================
# -- Control & Decision Parameters --------------------------------------------
# ==============================================================================
# Longitudinal Control (PID)
TARGET_SPEED = 20.0
K_P_SPEED = 1.0
K_I_SPEED = 0.0
K_D_SPEED = 0.1

# Lateral Control (PID)
K_P_STEER = 0.5
K_I_STEER = 0.0
K_D_STEER = 0.05

# Decision Making
DANGER_THRESHOLD_HEIGHT = 0.35
OBSTACLE_CLASSES = ['car', 'truck', 'bus', 'person', 'bicycle', 'motorcycle']