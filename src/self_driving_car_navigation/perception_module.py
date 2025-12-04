import torch
import torch.nn as nn

class PerceptionModule(nn.Module):
    def __init__(self):
        super(PerceptionModule, self).__init__()
        # 图像特征提取
        self.image_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # (3,128,128)→(32,64,64)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (32,64,64)→(64,32,32)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (64,32,32)→(128,16,16)
            nn.ReLU(),
            nn.Flatten()  # 128*16*16=32768
        )
        
        # 激光雷达特征提取（360度距离）
        self.lidar_cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),  # (1,360)→(64,360)
            nn.ReLU(),
            nn.MaxPool1d(2),  # (64,360)→(64,180)
            nn.Conv1d(64, 128, kernel_size=5, padding=2),  # (64,180)→(128,180)
            nn.ReLU(),
            nn.Flatten()  # 128*180=23040
        )

    def forward(self, imu_data, image, lidar_data):
        # 图像特征
        image_features = self.image_cnn(image)  # (batch, 32768)
        
        # 激光雷达特征
        lidar_features = self.lidar_cnn(lidar_data)  # (batch, 23040)
        
        # 场景信息（融合图像和激光雷达）
        scene_info = torch.cat([image_features, lidar_features], dim=1)  # (batch, 55808)
        
        # 语义分割（简化为图像特征）
        segmentation = image_features
        
        # 里程计（IMU数据）
        odometry = imu_data
        
        # 障碍物特征（激光雷达）
        obstacles = lidar_features
        
        # 边界特征（最大/最小距离）
        boundary = torch.cat([
            lidar_data.min(dim=2)[0],
            lidar_data.max(dim=2)[0]
        ], dim=1)  # (batch, 2)
        
        return scene_info, segmentation, odometry, obstacles, boundary