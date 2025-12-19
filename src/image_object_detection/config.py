# config.py
# 功能：定义项目全局配置参数，便于统一管理和修改

import os


class Config:
    """
    应用程序的配置类。
    所有核心参数（如模型路径、默认图像路径、阈值等）集中在此初始化，
    便于维护和部署时调整。
    """

    def __init__(self):
        # 获取当前 config.py 文件所在目录的绝对路径
        # 使用 os.path.abspath 确保路径在不同操作系统下一致
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # 默认测试图像路径：位于项目根目录下的 data/test.jpg
        # 若用户未指定图像路径，系统将尝试加载此文件
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.default_image_path = os.path.join(base_dir, "data", "test.jpg")  #注意：项目根目录下的data文件!!!

        
        # YOLO 模型权重文件路径
        # 支持：
        #   - 内置模型名（如 "yolov8n.pt"，首次运行会自动下载）
        #   - 自定义训练模型的相对或绝对路径（如 "runs/detect/train/weights/best.pt"）
        self.model_path = "yolov8n.pt"

        # 目标检测的置信度阈值
        # 只有置信度 ≥ 此值的检测框才会被保留和显示
        # 范围：0.0 ~ 1.0，值越高，结果越严格
        self.confidence_threshold = 0.35

        # 摄像头设备索引
        # 通常 0 表示内置摄像头，外接摄像头可能为 1、2 等
        self.camera_index = 0

        # FPS（帧率）输出的时间间隔（单位：秒）
        # 例如设为 1.0 表示每秒打印一次当前处理速度
        self.output_interval = 1.0


