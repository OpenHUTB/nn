import torch
import torch.nn as nn

class PerceptionModule(nn.Module):
    def __init__(self):
        super(PerceptionModule, self).__init__()

        # 原有语义分割网络保持不变
        self.segmentation_net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),

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
        
        # 修正：激光雷达障碍物检测子网络（输入通道从256改为1，适配实际激光雷达通道数）
        self.obstacle_net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1),  # 输入通道=1（激光雷达数据通道数）
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=16),  # 固定输出长度
            nn.Flatten()  # 展平为特征向量
        )

    def forward(self, imu_data, image, lidar_data):

        # 原有视觉特征提取保持不变
        segmentation = self.segmentation_net(image)
        scene_info = segmentation.mean(dim=(2, 3))  # 视觉场景信息
        odometry = imu_data  # IMU里程计数据
        boundary = lidar_data.max(dim=1)[0]  # 边界检测保持不变
        
        # 关键修改：将激光雷达4D数据[batch, 1, 64, 64]转为3D[batch, 1, 64*64]，适配1D卷积
        # 展平后两个空间维度（64x64 → 4096）
        lidar_reshaped = lidar_data.flatten(start_dim=2)  # 形状变为[batch, 1, 64*64=4096]
        
        # 用专用子网络处理激光雷达数据，提取障碍物特征
        obstacles = self.obstacle_net(lidar_reshaped)  # 输出[batch, 64*16=1024]

        
        return scene_info, segmentation, odometry, obstacles, boundary