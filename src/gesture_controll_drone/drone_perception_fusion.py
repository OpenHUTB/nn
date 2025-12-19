"""
无人机感知融合模块
融合多种传感器数据，提供环境感知能力
"""

import numpy as np
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import math

class SensorType(Enum):
    """传感器类型枚举"""
    CAMERA = "camera"
    LIDAR = "lidar"
    RADAR = "radar"
    GPS = "gps"
    IMU = "imu"
    SONAR = "sonar"

@dataclass
class SensorData:
    """传感器数据结构"""
    sensor_type: SensorType
    timestamp: float
    data: np.ndarray
    confidence: float = 1.0
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

class ObjectDetection:
    """检测到的物体类"""
    def __init__(self, obj_id: int, obj_type: str, position: Tuple[float, float, float],
                 confidence: float, size: Tuple[float, float, float], velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self.id = obj_id
        self.type = obj_type
        self.position = np.array(position)
        self.confidence = confidence
        self.size = size
        self.velocity = np.array(velocity)
        self.last_seen = time.time()
        self.track_history = [self.position.copy()]
        
    def update(self, new_position: Tuple[float, float, float], confidence: float = 1.0):
        """更新物体位置"""
        old_position = self.position.copy()
        self.position = np.array(new_position)
        self.velocity = (self.position - old_position) / (time.time() - self.last_seen)
        self.last_seen = time.time()
        self.confidence = confidence
        self.track_history.append(self.position.copy())
        # 保持最近20个轨迹点
        if len(self.track_history) > 20:
            self.track_history.pop(0)
    
    def predict_position(self, time_ahead: float = 0.1) -> np.ndarray:
        """预测未来位置"""
        return self.position + self.velocity * time_ahead
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "type": self.type,
            "position": self.position.tolist(),
            "confidence": self.confidence,
            "size": self.size,
            "velocity": self.velocity.tolist()
        }

class PerceptionFusionSystem:
    """
    多传感器融合感知系统
    融合摄像头、激光雷达、雷达等数据，构建环境模型
    """
    
    def __init__(self, fusion_method: str = "kalman"):
        """
        初始化感知融合系统
        
        参数:
            fusion_method: 融合方法 ("kalman", "particle", "bayesian")
        """
        self.fusion_method = fusion_method
        self.sensor_data = {}
        self.detected_objects = {}
        self.object_id_counter = 0
        self.fusion_confidence = 0.8
        self.update_rate = 10  # Hz
        
        # 传感器权重配置
        self.sensor_weights = {
            SensorType.CAMERA: 0.4,
            SensorType.LIDAR: 0.3,
            SensorType.RADAR: 0.2,
            SensorType.SONAR: 0.1
        }
        
        # 初始化卡尔曼滤波器（如果使用）
        if fusion_method == "kalman":
            self._init_kalman_filters()
        
        print(f"🤖 感知融合系统初始化完成，使用 {fusion_method} 融合方法")
    
    def _init_kalman_filters(self):
        """初始化卡尔曼滤波器"""
        self.kalman_filters = {}
        print("✅ 卡尔曼滤波器已初始化")
    
    def add_sensor_data(self, sensor_data: SensorData):
        """
        添加传感器数据
        
        参数:
            sensor_data: 传感器数据
        """
        self.sensor_data[sensor_data.sensor_type] = sensor_data
        
        # 数据融合处理
        if len(self.sensor_data) >= 2:  # 至少有2个传感器数据时进行融合
            self._fuse_sensor_data()
    
    def _fuse_sensor_data(self):
        """融合多传感器数据"""
        fused_objects = []
        
        # 按传感器类型分别处理
        camera_objects = self._process_camera_data()
        lidar_objects = self._process_lidar_data()
        radar_objects = self._process_radar_data()
        
        # 对象关联与融合
        all_objects = camera_objects + lidar_objects + radar_objects
        
        for obj in all_objects:
            # 查找是否有匹配的现有物体
            matched_obj = self._associate_object(obj)
            
            if matched_obj:
                # 更新现有物体
                matched_obj.update(obj.position, obj.confidence)
            else:
                # 创建新物体
                self.object_id_counter += 1
                new_obj = ObjectDetection(
                    self.object_id_counter,
                    obj.type,
                    obj.position,
                    obj.confidence,
                    obj.size,
                    obj.velocity
                )
                self.detected_objects[self.object_id_counter] = new_obj
                fused_objects.append(new_obj)
        
        return fused_objects
    
    def _process_camera_data(self) -> List[ObjectDetection]:
        """处理摄像头数据（模拟）"""
        if SensorType.CAMERA not in self.sensor_data:
            return []
        
        data = self.sensor_data[SensorType.CAMERA]
        
        # 模拟物体检测
        objects = []
        if data.confidence > 0.5:
            # 模拟检测到障碍物
            obstacle = ObjectDetection(
                0, "obstacle",
                (data.position[0] + 1.0, data.position[1] + 0.5, 2.0),
                0.8, (0.5, 0.5, 1.0)
            )
            objects.append(obstacle)
        
        return objects
    
    def _process_lidar_data(self) -> List[ObjectDetection]:
        """处理激光雷达数据（模拟）"""
        if SensorType.LIDAR not in self.sensor_data:
            return []
        
        data = self.sensor_data[SensorType.LIDAR]
        
        # 模拟点云处理
        objects = []
        points = data.data if data.data.size > 0 else np.random.randn(10, 3) * 2
        
        for i, point in enumerate(points[:3]):  # 模拟前3个点作为物体
            obj = ObjectDetection(
                0, "lidar_object",
                tuple(point),
                0.9, (0.3, 0.3, 0.3)
            )
            objects.append(obj)
        
        return objects
    
    def _process_radar_data(self) -> List[ObjectDetection]:
        """处理雷达数据（模拟）"""
        if SensorType.RADAR not in self.sensor_data:
            return []
        
        data = self.sensor_data[SensorType.RADAR]
        
        # 模拟雷达目标检测
        objects = []
        if data.confidence > 0.6:
            # 模拟动态目标
            dynamic_obj = ObjectDetection(
                0, "dynamic_object",
                (data.position[0] + 0.8, data.position[1] + 1.2, 1.5),
                0.85, (0.4, 0.4, 0.4),
                velocity=(0.5, 0.2, 0.0)
            )
            objects.append(dynamic_obj)
        
        return objects
    
    def _associate_object(self, new_obj: ObjectDetection) -> Optional[ObjectDetection]:
        """
        关联新检测到的物体与现有物体
        
        参数:
            new_obj: 新检测到的物体
            
        返回:
            匹配的现有物体，如果没有则返回None
        """
        for obj_id, existing_obj in self.detected_objects.items():
            # 计算距离
            distance = np.linalg.norm(new_obj.position - existing_obj.position)
            
            # 距离阈值和类型匹配
            if distance < 2.0 and new_obj.type == existing_obj.type:
                # 更新物体
                return existing_obj
        
        return None
    
    def get_environment_map(self) -> Dict:
        """
        获取环境地图
        
        返回:
            包含所有检测物体的环境地图
        """
        env_map = {
            "timestamp": time.time(),
            "object_count": len(self.detected_objects),
            "objects": [obj.to_dict() for obj in self.detected_objects.values()],
            "hazards": self._detect_hazards(),
            "free_space": self._calculate_free_space()
        }
        
        return env_map
    
    def _detect_hazards(self) -> List[Dict]:
        """检测危险区域"""
        hazards = []
        
        for obj_id, obj in self.detected_objects.items():
            # 判断是否为危险物体（靠近无人机）
            distance_to_drone = np.linalg.norm(obj.position)
            
            if distance_to_drone < 5.0:  # 5米内视为危险
                hazard = {
                    "id": obj_id,
                    "type": obj.type,
                    "position": obj.position.tolist(),
                    "distance": float(distance_to_drone),
                    "threat_level": "high" if distance_to_drone < 2.0 else "medium"
                }
                hazards.append(hazard)
        
        return hazards
    
    def _calculate_free_space(self) -> Dict:
        """计算自由空间"""
        return {
            "estimated_area": 100.0,  # 平方米
            "clearance_height": 10.0,  # 米
            "safe_directions": ["north", "east", "up"]
        }
    
    def predict_collisions(self, drone_position: Tuple[float, float, float], 
                          drone_velocity: Tuple[float, float, float],
                          time_horizon: float = 3.0) -> List[Dict]:
        """
        预测碰撞
        
        参数:
            drone_position: 无人机当前位置
            drone_velocity: 无人机当前速度
            time_horizon: 预测时间范围（秒）
            
        返回:
            碰撞预测列表
        """
        collisions = []
        drone_pos = np.array(drone_position)
        drone_vel = np.array(drone_velocity)
        
        for obj_id, obj in self.detected_objects.items():
            # 预测未来位置
            obj_future_pos = obj.predict_position(time_horizon)
            drone_future_pos = drone_pos + drone_vel * time_horizon
            
            # 计算最小距离
            min_distance = float('inf')
            time_to_collision = float('inf')
            
            # 简单的线性预测碰撞检测
            for t in np.linspace(0, time_horizon, 30):
                obj_pos_t = obj.position + obj.velocity * t
                drone_pos_t = drone_pos + drone_vel * t
                distance = np.linalg.norm(obj_pos_t - drone_pos_t)
                
                if distance < min_distance:
                    min_distance = distance
                    time_to_collision = t
            
            # 检查是否可能碰撞
            if min_distance < 1.0:  # 1米内视为可能碰撞
                collision = {
                    "object_id": obj_id,
                    "object_type": obj.type,
                    "time_to_collision": time_to_collision,
                    "min_distance": min_distance,
                    "recommended_action": self._get_avoidance_action(drone_vel, obj.velocity)
                }
                collisions.append(collision)
        
        return collisions
    
    def _get_avoidance_action(self, drone_vel: np.ndarray, obj_vel: np.ndarray) -> str:
        """获取避障动作建议"""
        relative_vel = obj_vel - drone_vel
        
        if relative_vel[0] > 0:
            return "move_left"
        elif relative_vel[0] < 0:
            return "move_right"
        elif relative_vel[1] > 0:
            return "move_down"
        elif relative_vel[1] < 0:
            return "move_up"
        else:
            return "hover"
    
    def export_environment_data(self, filename: str = "environment_map.json"):
        """导出环境数据到JSON文件"""
        env_data = self.get_environment_map()
        
        with open(filename, 'w') as f:
            json.dump(env_data, f, indent=2, default=str)
        
        print(f"✅ 环境数据已导出到 {filename}")
        return filename

# 使用示例
def demo_perception_fusion():
    """演示感知融合系统的使用"""
    print("🚀 开始感知融合演示...")
    
    # 创建感知融合系统
    fusion_system = PerceptionFusionSystem(fusion_method="kalman")
    
    # 模拟传感器数据
    print("\n📡 模拟传感器数据输入...")
    
    # 摄像头数据
    camera_data = SensorData(
        sensor_type=SensorType.CAMERA,
        timestamp=time.time(),
        data=np.random.rand(100, 3),  # 模拟100个特征点
        confidence=0.85,
        position=(0.0, 0.0, 0.0)
    )
    fusion_system.add_sensor_data(camera_data)
    
    # 激光雷达数据
    lidar_data = SensorData(
        sensor_type=SensorType.LIDAR,
        timestamp=time.time(),
        data=np.random.rand(500, 3) * 10,  # 模拟500个点云
        confidence=0.92,
        position=(0.1, 0.1, 0.0)
    )
    fusion_system.add_sensor_data(lidar_data)
    
    # 雷达数据
    radar_data = SensorData(
        sensor_type=SensorType.RADAR,
        timestamp=time.time(),
        data=np.random.rand(10, 4),  # 模拟10个雷达目标
        confidence=0.78,
        position=(0.05, -0.05, 0.0)
    )
    fusion_system.add_sensor_data(radar_data)
    
    # 获取环境地图
    print("\n🗺️  生成环境地图...")
    env_map = fusion_system.get_environment_map()
    print(f"检测到 {env_map['object_count']} 个物体")
    
    # 碰撞预测
    print("\n⚠️  碰撞预测分析...")
    collisions = fusion_system.predict_collisions(
        drone_position=(0.0, 0.0, 1.0),
        drone_velocity=(1.0, 0.0, 0.0),
        time_horizon=2.0
    )
    
    if collisions:
        print(f"预测到 {len(collisions)} 个潜在碰撞:")
        for collision in collisions:
            print(f"  物体 {collision['object_id']}: {collision['object_type']}, "
                  f"碰撞时间: {collision['time_to_collision']:.2f}s, "
                  f"动作: {collision['recommended_action']}")
    else:
        print("✅ 无碰撞风险")
    
    # 导出数据
    fusion_system.export_environment_data()
    
    print("\n✅ 感知融合演示完成")
    return fusion_system

if __name__ == "__main__":
    demo_perception_fusion()
