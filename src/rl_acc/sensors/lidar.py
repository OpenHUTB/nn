"""
激光雷达传感器模块 - Lidar Sensor
实现激光雷达目标检测
"""

import numpy as np
import time
from typing import List, Optional
from .sensor_fusion import SensorData


class LidarSensor:
    """
    激光雷达传感器
    模拟固态/机械式激光雷达的目标检测功能
    """

    def __init__(self,
                 num_channels: int = 128,  # 通道数
                 horizontal_fov: float = 120.0,  # 水平视野（度）
                 vertical_fov: float = 40.0,  # 垂直视野（度）
                 max_range: float = 250.0,  # 最大检测距离（米）
                 min_range: float = 0.5,  # 最小检测距离（米）
                 angular_resolution: float = 0.2):  # 角度分辨率（度）
        """
        初始化激光雷达传感器

        Args:
            num_channels: 激光雷达通道数
            horizontal_fov: 水平视野（度）
            vertical_fov: 垂直视野（度）
            max_range: 最大检测距离（米）
            min_range: 最小检测距离（米）
            angular_resolution: 角度分辨率（度）
        """
        self.num_channels = num_channels
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov
        self.max_range = max_range
        self.min_range = min_range
        self.angular_resolution = angular_resolution

    def detect(self,
              ego_speed: float,
              front_objects: List[dict]) -> List[SensorData]:
        """
        执行激光雷达检测

        Args:
            ego_speed: 自车速度（米/秒）
            front_objects: 前方目标列表，每个目标包含:
                - distance: 距离（米）
                - speed: 目标速度（米/秒）
                - azimuth: 方位角（度）

        Returns:
            检测到的目标列表
        """
        detections = []
        current_time = time.time()

        for obj in front_objects:
            distance = obj['distance']
            target_speed = obj['speed']
            azimuth_deg = obj.get('azimuth', 0.0)
            azimuth_rad = np.radians(azimuth_deg)

            # 过滤：超出检测范围
            if distance > self.max_range or distance < self.min_range:
                continue

            # 过滤：超出水平视野范围
            if abs(azimuth_deg) > self.horizontal_fov / 2:
                continue

            # 计算相对速度
            relative_speed = target_speed - ego_speed

            # 计算置信度（基于点云密度）
            confidence = self._calculate_confidence(distance)

            # 添加检测结果
            detection = SensorData(
                sensor_type='lidar',
                timestamp=current_time,
                distance=distance,
                relative_speed=relative_speed,
                azimuth=azimuth_rad,
                confidence=confidence
            )
            detections.append(detection)

        return detections

    def _calculate_confidence(self, distance: float) -> float:
        """
        计算检测置信度

        Args:
            distance: 目标距离（米）

        Returns:
            置信度 [0, 1]
        """
        # 激光雷达在中等距离表现最好
        if distance < 50:
            confidence = 0.9
        elif distance < 100:
            confidence = 0.85
        elif distance < 150:
            confidence = 0.8
        else:
            # 远距离点云稀疏，置信度下降
            confidence = 0.75 - (distance - 150) * 0.002

        return max(0.6, min(confidence, 0.95))
