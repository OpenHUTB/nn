"""
传感器融合模块 - Sensor Fusion Module
实现多传感器数据融合，包括雷达、摄像头、激光雷达等
用于提高目标检测的准确性和可靠性
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SensorData:
    """传感器数据结构"""
    sensor_type: str  # 'radar', 'camera', 'lidar'
    timestamp: float
    distance: float  # 距离（米）
    relative_speed: float  # 相对速度（米/秒）
    azimuth: float  # 方位角（弧度）
    confidence: float  # 置信度 [0, 1]
    object_id: Optional[int] = None  # 目标ID


@dataclass
class FusedObject:
    """融合后的目标数据结构"""
    object_id: int
    distance: float
    relative_speed: float
    azimuth: float
    confidence: float
    data_sources: List[str]  # 数据来源列表


class SensorFusion:
    """
    传感器融合器
    使用加权融合算法整合多个传感器的检测结果
    """

    def __init__(self,
                 radar_weight: float = 0.4,
                 camera_weight: float = 0.3,
                 lidar_weight: float = 0.3,
                 fusion_threshold: float = 0.5,
                 max_distance: float = 200.0):
        """
        初始化传感器融合器

        Args:
            radar_weight: 雷达权重
            camera_weight: 摄像头权重
            lidar_weight: 激光雷达权重
            fusion_threshold: 融合阈值，低于此置信度不输出
            max_distance: 最大检测距离（米）
        """
        self.radar_weight = radar_weight
        self.camera_weight = camera_weight
        self.lidar_weight = lidar_weight
        self.fusion_threshold = fusion_threshold
        self.max_distance = max_distance
        self.object_counter = 0
        self.tracked_objects: Dict[int, FusedObject] = {}

    def fuse_sensors(self,
                    radar_data: List[SensorData],
                    camera_data: List[SensorData],
                    lidar_data: List[SensorData]) -> List[FusedObject]:
        """
        融合多传感器数据

        Args:
            radar_data: 雷达检测数据
            camera_data: 摄像头检测数据
            lidar_data: 激光雷达检测数据

        Returns:
            融合后的目标列表
        """
        # 合并所有传感器数据
        all_data = []
        all_data.extend([(d, 'radar') for d in radar_data])
        all_data.extend([(d, 'camera') for d in camera_data])
        all_data.extend([(d, 'lidar') for d in lidar_data])

        if not all_data:
            return []

        # 按距离分组
        clusters = self._cluster_by_distance(all_data, distance_threshold=5.0)

        # 融合每个簇
        fused_objects = []
        for cluster in clusters:
            fused_obj = self._fuse_cluster(cluster)
            if fused_obj and fused_obj.confidence >= self.fusion_threshold:
                fused_objects.append(fused_obj)

        return fused_objects

    def _cluster_by_distance(self,
                            data: List[Tuple[SensorData, str]],
                            distance_threshold: float = 5.0) -> List[List[Tuple[SensorData, str]]]:
        """
        按距离对检测结果进行聚类

        Args:
            data: 传感器数据列表
            distance_threshold: 距离阈值（米）

        Returns:
            聚类结果
        """
        if not data:
            return []

        clusters = []
        used = set()

        for i, (sensor_data, sensor_type) in enumerate(data):
            if i in used:
                continue

            cluster = [(sensor_data, sensor_type)]
            used.add(i)

            for j, (other_data, other_type) in enumerate(data):
                if j in used:
                    continue

                # 计算距离差
                distance_diff = abs(sensor_data.distance - other_data.distance)
                if distance_diff <= distance_threshold:
                    cluster.append((other_data, other_type))
                    used.add(j)

            clusters.append(cluster)

        return clusters

    def _fuse_cluster(self, cluster: List[Tuple[SensorData, str]]) -> Optional[FusedObject]:
        """
        融合一个簇内的数据

        Args:
            cluster: 簇内数据列表

        Returns:
            融合后的目标
        """
        if not cluster:
            return None

        # 提取数据
        distances = []
        speeds = []
        azimuths = []
        confidences = []
        sources = []
        weights = []

        for sensor_data, sensor_type in cluster:
            distances.append(sensor_data.distance)
            speeds.append(sensor_data.relative_speed)
            azimuths.append(sensor_data.azimuth)
            confidences.append(sensor_data.confidence)
            sources.append(sensor_type)

            # 获取对应传感器的权重
            if sensor_type == 'radar':
                weights.append(self.radar_weight)
            elif sensor_type == 'camera':
                weights.append(self.camera_weight)
            else:  # lidar
                weights.append(self.lidar_weight)

        # 归一化权重
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)

        # 加权平均
        fused_distance = sum(d * w for d, w in zip(distances, weights))
        fused_speed = sum(s * w for s, w in zip(speeds, weights))
        fused_azimuth = sum(a * w for a, w in zip(azimuths, weights))
        fused_confidence = sum(c * w for c, w in zip(confidences, weights))

        # 更新跟踪对象
        self.object_counter += 1
        fused_object = FusedObject(
            object_id=self.object_counter,
            distance=fused_distance,
            relative_speed=fused_speed,
            azimuth=fused_azimuth,
            confidence=fused_confidence,
            data_sources=sources
        )

        return fused_object

    def update_tracking(self, fused_objects: List[FusedObject]) -> Dict[int, FusedObject]:
        """
        更新目标跟踪

        Args:
            fused_objects: 当前帧融合结果

        Returns:
            跟踪的目标字典
        """
        # 简单跟踪：更新现有目标或添加新目标
        for obj in fused_objects:
            self.tracked_objects[obj.object_id] = obj

        return self.tracked_objects

    def get_closest_object(self) -> Optional[FusedObject]:
        """
        获取最近的目标

        Returns:
            最近目标，如果没有则返回None
        """
        if not self.tracked_objects:
            return None

        closest = None
        min_distance = float('inf')

        for obj in self.tracked_objects.values():
            if obj.distance < min_distance:
                min_distance = obj.distance
                closest = obj

        return closest

    def get_TTC(self, ego_speed: float) -> Optional[float]:
        """
        计算与最近目标的碰撞时间（Time To Collision）

        Args:
            ego_speed: 自车速度（米/秒）

        Returns:
            TTC（秒），如果无法计算则返回None
        """
        closest = self.get_closest_object()
        if not closest:
            return None

        # TTC = 距离 / 相对速度
        relative_speed = closest.relative_speed
        if relative_speed <= 0:  # 目标在远离或静止
            return None

        TTC = closest.distance / relative_speed
        return TTC
