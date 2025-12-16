# config.py
class Config:
    def __init__(self):
        # 测试图像路径
        self.test_image_path = r"C:\Users\apple\OneDrive\桌面\test.jpg"
        
        # YOLO模型配置
        self.model_path = "yolov8n.pt"
        self.confidence_threshold = 0.25
        
        # 摄像头配置
        self.camera_index = 0
        
        # 输出频率控制（秒）
        self.output_interval = 1.0