# -*- coding: utf-8 -*-
"""
项目全局配置模块
包含 CARLA 服务器连接设置、传感器参数以及 YOLO 模型路径
"""

# CARLA 服务器设置
CARLA_HOST = '127.0.0.1'
CARLA_PORT = 2000
TIMEOUT = 10.0

# 传感器设置
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600
CAMERA_FOV = 90

# 自动驾驶主车设置
VEHICLE_FILTER = 'vehicle.tesla.model3'  # 默认生成特斯拉 Model 3
SPAWN_POINT_INDEX = 1                    # 默认出生点索引

# YOLO 模型配置 (预留)
MODEL_CFG = "models/yolov3.cfg"
MODEL_WEIGHTS = "models/yolov3.weights"
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# 路径设置
OUTPUT_PATH = "output/"

# --- 新增：车辆控制参数 (PID Controller) ---
# 纵向控制 (速度)
TARGET_SPEED = 20.0  # 目标时速 km/h
K_P_SPEED = 1.0      # 比例系数
K_I_SPEED = 0.0      # 积分系数
K_D_SPEED = 0.1      # 微分系数

# 横向控制 (转向 - 预留)
K_P_STEER = 0.5
K_I_STEER = 0.0
K_D_STEER = 0.0