import torch
import torch.nn as nn

class PerceptionModule(nn.Module):
    def __init__(self):
        super(PerceptionModule, self).__init__()
        self.segmentation_net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, imu_data, image, lidar_data):
        segmentation = self.segmentation_net(image)
        scene_info = segmentation.mean(dim=(2, 3))  # 视觉信息提取
        odometry = imu_data  # IMU用于里程计
        obstacles = lidar_data.mean(dim=1)  # 障碍物检测
        boundary = lidar_data.max(dim=1)[0]  # 边界检测

        return scene_info, segmentation, odometry, obstacles, boundary
