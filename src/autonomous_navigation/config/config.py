import os
import yaml

# 基础配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
    "model_path": os.path.join(BASE_DIR, "models/anomaly_detector.pth"),
    "confidence_threshold": 0.7,
    "fusion_method": "weighted",  # 多模态融合方式：weighted/concat/attention
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def load_config(custom_config_path=None):
    """加载自定义配置文件（可选）"""
    if custom_config_path and os.path.exists(custom_config_path):
        with open(custom_config_path, "r") as f:
            custom_config = yaml.safe_load(f)
        # 合并配置
        globals().update(custom_config)
    return globals()