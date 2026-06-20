"""
雷达传感器模块 - Radar Sensor
实现毫米波雷达目标检测
"""

import numpy as np
import time
from typing import List, Optional
from .sensor_fusion import SensorData


class RadarSensor:
    """
    雷达传感器
    模拟毫米波雷达的目标检测功能
    """

    def __init__(self,
                 field_of_view: float = 60.0,  # 水平视野（度）
                 max_range: float = 200.0,  # 最大检测距离（米）
                 range_resolution: float = 0.5,  # 距离分辨率（米）
                 max_tracks: int = 32):  # 最大跟踪目标数
        """
        初始化雷达传感器

        Args:
            field_of_view: 水平视野（度）
            max_range: 最大检测距离（米）
            range_resolution: 距离分辨率（米）
            max_tracks: 最大跟踪目标数
        """
        self.field_of_view = field_of_view
        self.max_range = max_range
        self.range_resolution = range_resolution
        self.max_tracks = max_tracks

    def detect(self,
              ego_speed: float,
              front_objects: List[dict]) -> List[SensorData]:
        """
        执行雷达检测

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
            if distance > self.max_range or distance < 0:
                continue

            # 过滤：超出视野范围
            if abs(azimuth_deg) > self.field_of_view / 2:
                continue

            # 计算相对速度
            relative_speed = target_speed - ego_speed

            # 计算置信度（基于距离和SNR模拟）
            confidence = self._calculate_confidence(distance)

            # 添加检测结果
            detection = SensorData(
                sensor_type='radar',
                timestamp=current_time,
                distance=distance,
                relative_speed=relative_speed,
                azimuth=azimuth_rad,
                confidence=confidence
            )
            detections.append(detection)

            # 限制最大跟踪数
            if len(detections) >= self.max_tracks:
                break

        return detections

    def _calculate_confidence(self, distance: float) -> float:
        """
        计算检测置信度

        Args:
            distance: 目标距离（米）

        Returns:
            置信度 [0, 1]
        """
        # 简单的置信度模型：距离越近，置信度越高
        if distance <= 0:
            return 0.0

        confidence = 1.0 - (distance / self.max_range) * 0.3
        confidence = np.clip(confidence, 0.5, 1.0)

        return confidence
