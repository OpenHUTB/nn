# scripts/models/perception_module.py
import torch
import numpy as np
from cv_bridge import CvBridge  # 若涉及图像转换，需提前安装：pip install cv-bridge

class PerceptionModule:
    def __init__(self, 
                 image_input_shape=(3, 128, 128),  # 图像输入维度：(通道数, 高, 宽)
                 lidar_input_dim=360,               # LiDAR输入维度（360个距离点）
                 imu_input_dim=6,                   # IMU输入维度（加速度x/y/z + 角速度x/y/z）
                 output_feature_dim=128):           # 输出感知特征维度
        """初始化感知模块"""
        self.image_shape = image_input_shape
        self.lidar_dim = lidar_input_dim
        self.imu_dim = imu_input_dim
        self.feature_dim = output_feature_dim
        
        # 示例：初始化一个简单的特征融合层（可替换为你的实际模型）
        self.fusion_layer = torch.nn.Linear(
            (image_input_shape[0]*image_input_shape[1]*image_input_shape[2]) + lidar_input_dim + imu_input_dim,
            output_feature_dim
        )
        self.bridge = CvBridge()  # 图像格式转换工具（ROS Image ↔ OpenCV）

    def process(self, imu_data, image_data, lidar_data):
        """
        处理传感器数据，输出融合特征
        参数：
            imu_data: torch.Tensor - IMU数据，shape=(B, 6)
            image_data: torch.Tensor - 图像数据，shape=(B, 3, 128, 128)
            lidar_data: torch.Tensor - LiDAR数据，shape=(B, 360)
        返回：
            fused_feature: torch.Tensor - 融合特征，shape=(B, 128)
        """
        # 1. 扁平化各传感器数据
        image_flat = image_data.flatten(start_dim=1)  # (B, 3*128*128)
        lidar_flat = lidar_data.flatten(start_dim=1)  # (B, 360)
        imu_flat = imu_data.flatten(start_dim=1)      # (B, 6)

        # 2. 拼接数据并融合
        combined_data = torch.cat([image_flat, lidar_flat, imu_flat], dim=1)
        fused_feature = self.fusion_layer(combined_data)  # 简单示例：全连接层融合

        return fused_feature

    def ros_image_to_tensor(self, ros_image_msg):
        """将ROS Image消息转换为模型输入的Tensor"""
        cv_image = self.bridge.imgmsg_to_cv2(ros_image_msg, "bgr8")  # ROS → OpenCV
        tensor_image = torch.tensor(cv_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 128, 128)
        return tensor_image