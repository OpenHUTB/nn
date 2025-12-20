import random
import math
import carla
import time
import json
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class VehicleState:
    """车辆状态信息"""
    vehicle_id: int
    type_id: str
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    timestamp: float
    sensors_available: List[str]


@dataclass
class V2XMessage:
    """V2X通信消息"""
    sender_id: int
    message_type: str  # 'state', 'object', 'warning', 'control'
    data: Dict
    timestamp: float
    ttl: float  # 生存时间
    priority: int  # 优先级


class MultiVehicleManager:
    """多车辆协同管理器"""

    def __init__(self, world, config, output_dir):
        self.world = world
        self.config = config
        self.output_dir = output_dir

        # 创建协同数据目录
        self.coop_dir = os.path.join(output_dir, "cooperative")
        os.makedirs(self.coop_dir, exist_ok=True)
        os.makedirs(os.path.join(self.coop_dir, "v2x_messages"), exist_ok=True)
        os.makedirs(os.path.join(self.coop_dir, "shared_perception"), exist_ok=True)

        # 车辆管理
        self.ego_vehicles = []  # 主车辆列表
        self.cooperative_vehicles = []  # 协同车辆列表
        self.vehicle_states = {}  # 车辆ID -> 车辆状态
        self.vehicle_sensors = {}  # 车辆ID -> 传感器列表

        # V2X通信
        self.v2x_messages = []
        self.communication_range = config.get('v2x', {}).get('communication_range', 300.0)  # 通信范围
        self.message_buffer = defaultdict(list)  # 车辆ID -> 接收的消息列表

        # 协同感知
        self.shared_objects = []  # 共享的感知对象
        self.fusion_results = {}  # 融合结果

        # 统计
        self.stats = {
            'total_messages': 0,
            'successful_transmissions': 0,
            'collaborative_detections': 0,
            'data_exchange_mb': 0.0
        }

    def spawn_cooperative_vehicles(self, num_vehicles: int = 3) -> List[carla.Actor]:
        """生成协同车辆"""
        blueprint_lib = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()

        if not spawn_points:
            print("警告：无生成点")
            return []

        # 车辆类型
        vehicle_types = [
            'vehicle.tesla.model3',
            'vehicle.audi.tt',
            'vehicle.nissan.patrol',
            'vehicle.bmw.grandtourer',
            'vehicle.mercedes.coupe'
        ]

        spawned_vehicles = []

        for i in range(min(num_vehicles, len(spawn_points))):
            try:
                # 选择车辆类型
                vtype = random.choice(vehicle_types)
                vehicle_bp = random.choice(blueprint_lib.filter(vtype))

                # 设置车辆属性
                vehicle_bp.set_attribute('role_name', f'coop_vehicle_{i}')

                # 选择生成点
                spawn_point = spawn_points[i % len(spawn_points)]

                # 调整位置，使车辆不在同一位置
                offset_x = random.uniform(-5.0, 5.0)
                offset_y = random.uniform(-5.0, 5.0)
                location = carla.Location(
                    x=spawn_point.location.x + offset_x,
                    y=spawn_point.location.y + offset_y,
                    z=spawn_point.location.z
                )

                # 轻微调整朝向
                rotation = carla.Rotation(
                    pitch=spawn_point.rotation.pitch,
                    yaw=spawn_point.rotation.yaw + random.uniform(-15, 15),
                    roll=spawn_point.rotation.roll
                )

                transform = carla.Transform(location, rotation)

                # 生成车辆
                vehicle = self.world.spawn_actor(vehicle_bp, transform)

                # 设置自动驾驶
                vehicle.set_autopilot(True)

                # 添加到列表
                self.cooperative_vehicles.append(vehicle)
                spawned_vehicles.append(vehicle)

                # 初始化车辆状态
                self.vehicle_states[vehicle.id] = VehicleState(
                    vehicle_id=vehicle.id,
                    type_id=vehicle.type_id,
                    position=(0, 0, 0),
                    velocity=(0, 0, 0),
                    rotation=(0, 0, 0),
                    timestamp=time.time(),
                    sensors_available=[]
                )

                print(f"协同车辆 {i + 1} 生成: {vehicle.type_id}")

            except Exception as e:
                print(f"生成协同车辆失败: {e}")

        return spawned_vehicles

    def update_vehicle_states(self):
        """更新所有车辆状态"""
        current_time = time.time()

        for vehicle in self.ego_vehicles + self.cooperative_vehicles:
            try:
                if vehicle.is_alive:
                    location = vehicle.get_location()
                    velocity = vehicle.get_velocity()
                    rotation = vehicle.get_transform().rotation

                    self.vehicle_states[vehicle.id] = VehicleState(
                        vehicle_id=vehicle.id,
                        type_id=vehicle.type_id,
                        position=(location.x, location.y, location.z),
                        velocity=(velocity.x, velocity.y, velocity.z),
                        rotation=(rotation.pitch, rotation.yaw, rotation.roll),
                        timestamp=current_time,
                        sensors_available=self.vehicle_sensors.get(vehicle.id, [])
                    )
            except:
                pass

    def create_v2x_message(self, sender_id: int, message_type: str, data: Dict,
                           priority: int = 1) -> V2XMessage:
        """创建V2X消息"""
        message = V2XMessage(
            sender_id=sender_id,
            message_type=message_type,
            data=data,
            timestamp=time.time(),
            ttl=5.0,  # 5秒生存时间
            priority=priority
        )

        self.v2x_messages.append(message)
        self.stats['total_messages'] += 1

        return message

    def broadcast_message(self, message: V2XMessage):
        """广播消息给范围内的车辆"""
        if message.sender_id not in self.vehicle_states:
            return

        sender_state = self.vehicle_states[message.sender_id]
        sender_pos = sender_state.position

        recipients = []

        # 检查所有车辆是否在通信范围内
        for vehicle in self.ego_vehicles + self.cooperative_vehicles:
            if vehicle.id != message.sender_id and vehicle.id in self.vehicle_states:
                receiver_state = self.vehicle_states[vehicle.id]
                receiver_pos = receiver_state.position

                # 计算距离
                distance = math.sqrt(
                    (sender_pos[0] - receiver_pos[0]) ** 2 +
                    (sender_pos[1] - receiver_pos[1]) ** 2 +
                    (sender_pos[2] - receiver_pos[2]) ** 2
                )

                if distance <= self.communication_range:
                    recipients.append(vehicle.id)

                    # 添加到接收者消息缓冲区
                    self.message_buffer[vehicle.id].append({
                        'message': message,
                        'receive_time': time.time(),
                        'signal_strength': 1.0 - (distance / self.communication_range)
                    })

        self.stats['successful_transmissions'] += len(recipients)

        # 保存消息
        self._save_v2x_message(message, recipients)

        return recipients

    def share_perception_data(self, vehicle_id: int, detected_objects: List[Dict]):
        """共享感知数据"""
        if not detected_objects:
            return

        # 创建感知消息
        perception_data = {
            'vehicle_id': vehicle_id,
            'timestamp': time.time(),
            'objects': detected_objects,
            'vehicle_state': asdict(
                self.vehicle_states.get(vehicle_id, VehicleState(0, '', (0, 0, 0), (0, 0, 0), (0, 0, 0), 0, [])))
        }

        message = self.create_v2x_message(
            vehicle_id,
            'perception',
            perception_data,
            priority=2  # 感知数据优先级较高
        )

        # 广播消息
        recipients = self.broadcast_message(message)

        # 融合共享的感知数据
        if recipients:
            self._fuse_shared_perception(vehicle_id, detected_objects, recipients)

        return message

    def share_traffic_warning(self, vehicle_id: int, warning_type: str,
                              location: Tuple[float, float, float],
                              severity: str = 'medium'):
        """共享交通警告"""
        warning_data = {
            'warning_type': warning_type,  # 'accident', 'congestion', 'hazard', 'construction'
            'location': location,
            'severity': severity,
            'timestamp': time.time(),
            'source_vehicle': vehicle_id
        }

        message = self.create_v2x_message(
            vehicle_id,
            'warning',
            warning_data,
            priority=3  # 警告消息优先级最高
        )

        self.broadcast_message(message)

        return message

    def _fuse_shared_perception(self, source_id: int, objects: List[Dict], recipients: List[int]):
        """融合共享的感知数据"""
        fused_objects = []

        for obj in objects:
            # 转换为全局坐标系（简化处理，实际需要坐标变换）
            global_obj = obj.copy()
            global_obj['source_vehicles'] = [source_id]
            global_obj['confidence'] = obj.get('confidence', 0.8)  # 降低置信度

            # 检查是否已有类似对象
            matched = False
            for existing_obj in self.shared_objects:
                # 简单的对象匹配（基于位置）
                if self._objects_match(global_obj, existing_obj):
                    # 更新现有对象
                    existing_obj['source_vehicles'].append(source_id)
                    existing_obj['confidence'] = min(1.0, existing_obj.get('confidence', 0) + 0.1)
                    existing_obj['update_time'] = time.time()
                    matched = True
                    break

            if not matched:
                global_obj['detection_time'] = time.time()
                fused_objects.append(global_obj)

        self.shared_objects.extend(fused_objects)

        # 清理旧对象
        current_time = time.time()
        self.shared_objects = [
            obj for obj in self.shared_objects
            if current_time - obj.get('detection_time', 0) < 10.0  # 保留10秒内的对象
        ]

        if fused_objects:
            self.stats['collaborative_detections'] += len(fused_objects)

    def _objects_match(self, obj1: Dict, obj2: Dict, distance_threshold: float = 5.0) -> bool:
        """判断两个对象是否匹配"""
        if obj1.get('class') != obj2.get('class'):
            return False

        # 计算位置距离
        pos1 = obj1.get('position', {'x': 0, 'y': 0, 'z': 0})
        pos2 = obj2.get('position', {'x': 0, 'y': 0, 'z': 0})

        distance = math.sqrt(
            (pos1['x'] - pos2['x']) ** 2 +
            (pos1['y'] - pos2['y']) ** 2 +
            (pos1['z'] - pos2['z']) ** 2
        )

        return distance < distance_threshold

    def get_shared_perception_for_vehicle(self, vehicle_id: int) -> List[Dict]:
        """获取车辆可用的共享感知数据"""
        shared_data = []

        for obj in self.shared_objects:
            # 检查对象是否在车辆视野内（简化）
            shared_data.append(obj)

        return shared_data

    def coordinate_maneuvers(self, maneuvers: List[Dict]):
        """协调多车辆机动"""
        # 分配优先级和时序
        coordinated = []

        for i, maneuver in enumerate(maneuvers):
            coordinated_maneuver = maneuver.copy()
            coordinated_maneuver['sequence'] = i
            coordinated_maneuver['start_time'] = time.time() + i * 2.0  # 间隔2秒
            coordinated.append(coordinated_maneuver)

            # 发送协调消息
            message = self.create_v2x_message(
                maneuver.get('vehicle_id', 0),
                'coordination',
                coordinated_maneuver,
                priority=2
            )

            self.broadcast_message(message)

        return coordinated

    def _save_v2x_message(self, message: V2XMessage, recipients: List[int]):
        """保存V2X消息到文件"""
        message_data = {
            'message': asdict(message),
            'recipients': recipients,
            'transmission_time': time.time()
        }

        filename = f"v2x_{int(time.time() * 1000)}_{message.sender_id}_{message.message_type}.json"
        filepath = os.path.join(self.coop_dir, "v2x_messages", filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(message_data, f, indent=2, ensure_ascii=False)

    def save_shared_perception(self, frame_num: int):
        """保存共享感知数据"""
        data = {
            'frame_id': frame_num,
            'timestamp': time.time(),
            'shared_objects': self.shared_objects,
            'active_vehicles': len(self.ego_vehicles + self.cooperative_vehicles),
            'stats': self.stats
        }

        filepath = os.path.join(self.coop_dir, "shared_perception", f"frame_{frame_num:06d}.json")

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def generate_summary(self):
        """生成协同摘要报告"""
        summary = {
            'total_vehicles': len(self.ego_vehicles) + len(self.cooperative_vehicles),
            'ego_vehicles': len(self.ego_vehicles),
            'cooperative_vehicles': len(self.cooperative_vehicles),
            'v2x_stats': self.stats,
            'shared_objects_count': len(self.shared_objects),
            'communication_range': self.communication_range,
            'average_messages_per_vehicle': self.stats['total_messages'] / max(1, len(self.ego_vehicles) + len(
                self.cooperative_vehicles))
        }

        filepath = os.path.join(self.coop_dir, "cooperative_summary.json")

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return summary

    def cleanup(self):
        """清理资源"""
        print("清理协同车辆...")

        for vehicle in self.cooperative_vehicles:
            try:
                if vehicle.is_alive:
                    vehicle.destroy()
            except:
                pass

        self.cooperative_vehicles.clear()
        self.ego_vehicles.clear()
        self.vehicle_states.clear()
        self.message_buffer.clear()