<<<<<<< HEAD
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
=======
from pathlib import Path

import torch
import yaml

# 基础配置
BASE_DIR = Path(__file__).resolve().parent
CARLA_HOST = "localhost"
CARLA_PORT = 2000
CARLA_TIMEOUT = 10.0

# 非结构化场景配置
UNSTRUCTURED_SCENES = {
    "town07": {"weather": "RAINY", "road_type": "rural", "anomaly_types": ["pothole", "fallen_tree", "construction"]},
    "town10": {"weather": "FOGGY", "road_type": "suburb", "anomaly_types": ["broken_road", "stray_vehicle", "pedestrian_violation"]}
}

# 传感器配置
SENSOR_CONFIG = {
    "rgb_camera": {"fov": 110, "width": 1280, "height": 720, "position": [0.8, 0.0, 1.7]},
    "lidar": {"channels": 32, "range": 50, "points_per_second": 500000, "position": [0.0, 0.0, 2.4]},
    "radar": {"range": 100, "points_per_second": 10000, "position": [0.5, 0.0, 1.5]},
    "imu": {"frequency": 100, "position": [0.0, 0.0, 0.0]}
}

# 异常检测模型配置
ANOMALY_DETECTOR_CONFIG = {
    "model_path": str(BASE_DIR / "models" / "anomaly_detector.pth"),
    "confidence_threshold": 0.7,
    "fusion_method": "weighted",  # 多模态融合方式：weighted/concat/attention
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


def _merge_config_value(current_value, custom_value):
    if isinstance(current_value, dict) and isinstance(custom_value, dict):
        merged_value = current_value.copy()
        merged_value.update(custom_value)
        return merged_value
    return custom_value


def load_config(custom_config_path=None):
    """加载自定义配置文件（可选）"""
    if custom_config_path:
        config_path = Path(custom_config_path)
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                custom_config = yaml.safe_load(f) or {}

            for key, value in custom_config.items():
                globals()[key] = _merge_config_value(globals().get(key), value)
    return globals()
>>>>>>> upstream/main
