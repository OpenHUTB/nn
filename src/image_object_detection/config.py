# config.py
import os

class Config:
    def __init__(self):
        # 获取当前脚本所在目录（更可靠）
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.default_image_path = os.path.join(base_dir, "data", "test.jpg")
        
        self.model_path = "yolov8n.pt"
        self.confidence_threshold = 0.25
        self.camera_index = 0
        self.output_interval = 1.0
