"""
摄像头传感器模块 - Camera Sensor
实现视觉目标检测
"""

import numpy as np
import time
from typing import List, Optional
from .sensor_fusion import SensorData


class CameraSensor:
    """
    摄像头传感器
    模拟单目摄像头的目标检测和距离估计功能
    """

    def __init__(self,
                 image_width: int = 1920,
                 image_height: int = 1080,
                 focal_length: float = 1000.0,  # 焦距（像素）
                 field_of_view: float = 120.0,  # 水平视野（度）
                 detection_range: float = 150.0):  # 检测范围（米）
        """
        初始化摄像头传感器

        Args:
            image_width: 图像宽度（像素）
            image_height: 图像高度（像素）
            focal_length: 焦距（像素）
            field_of_view: 水平视野（度）
            detection_range: 检测范围（米）
        """
        self.image_width = image_width
        self.image_height = image_height
        self.focal_length = focal_length
        self.field_of_view = field_of_view
        self.detection_range = detection_range

    def detect(self,
              ego_speed: float,
              front_objects: List[dict]) -> List[SensorData]:
        """
        执行视觉检测

        Args:
            ego_speed: 自车速度（米/秒）
            front_objects: 前方目标列表，每个目标包含:
                - distance: 距离（米）
                - speed: 目标速度（米/秒）
                - azimuth: 方位角（度）
                - bbox_width: 边界框宽度（像素）- 可选

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
            bbox_width = obj.get('bbox_width', None)

            # 过滤：超出检测范围
            if distance > self.detection_range or distance < 0:
                continue

            # 过滤：超出视野范围
            if abs(azimuth_deg) > self.field_of_view / 2:
                continue

            # 计算相对速度
            relative_speed = target_speed - ego_speed

            # 计算置信度（基于距离和检测质量）
            confidence = self._calculate_confidence(distance, bbox_width)

            # 添加检测结果
            detection = SensorData(
                sensor_type='camera',
                timestamp=current_time,
                distance=distance,
                relative_speed=relative_speed,
                azimuth=azimuth_rad,
                confidence=confidence
            )
            detections.append(detection)

        return detections

    def _calculate_confidence(self,
                             distance: float,
                             bbox_width: Optional[float] = None) -> float:
        """
        计算检测置信度

        Args:
            distance: 目标距离（米）
            bbox_width: 边界框宽度（像素）

        Returns:
            置信度 [0, 1]
        """
        # 基础置信度
        confidence = 0.7

        # 距离惩罚（摄像头对近距离目标检测更准确）
        if distance < 50:
            confidence = 0.85
        elif distance < 100:
            confidence = 0.75
        else:
            confidence = 0.65

        # 如果有bbox信息，可以提高置信度
        if bbox_width is not None and bbox_width > 50:
            confidence += 0.1

        return min(confidence, 1.0)

    def estimate_distance_from_bbox(self,
                                   bbox_width: float,
                                   real_width: float = 4.5) -> float:
        """
        从边界框估计距离（单目测距）

        Args:
            bbox_width: 边界框宽度（像素）
            real_width: 真实物体宽度（米），默认小车约4.5米

        Returns:
            估计的距离（米）
        """
        if bbox_width <= 0:
            return 0.0

        # 简单针孔相机模型
        distance = (real_width * self.focal_length) / bbox_width
        return distance
