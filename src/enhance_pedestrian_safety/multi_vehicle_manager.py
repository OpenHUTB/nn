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

        # 创建子目录
        v2x_messages_dir = os.path.join(self.coop_dir, "v2x_messages")
        os.makedirs(v2x_messages_dir, exist_ok=True)

        shared_perception_dir = os.path.join(self.coop_dir, "shared_perception")
        os.makedirs(shared_perception_dir, exist_ok=True)

        # 车辆管理
        self.ego_vehicles = []  # 主车辆列表
        self.cooperative_vehicles = []  # 协同车辆列表
        self.vehicle_states = {}  # 车辆ID -> 车辆状态
        self.vehicle_sensors = {}  # 车辆ID -> 传感器列表

        # V2X通信
        self.v2x_messages = []
        self.communication_range = config.get('v2x', {}).get('communication_range', 300.0)
        self.message_buffer = defaultdict(list)

        # 协同感知
        self.shared_objects = []  # 共享的感知对象
        self.fusion_results = {}  # 融合结果

        # 统计
        self.stats = {
            'total_messages': 0,
            'successful_transmissions': 0,
            'collaborative_detections': 0,
            'data_exchange_mb': 0.0,
            'pedestrian_warnings_sent': 0,
            'safety_alerts': 0
        }

    def spawn_cooperative_vehicles(self, num_vehicles: int = 3) -> List[carla.Actor]:
        """生成协同车辆"""
        blueprint_lib = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()

        if not spawn_points:
            print("警告：无生成点")
            return []

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
                vtype = random.choice(vehicle_types)
                vehicle_bp = random.choice(blueprint_lib.filter(vtype))

                vehicle_bp.set_attribute('role_name', f'coop_vehicle_{i}')

                spawn_point = spawn_points[i % len(spawn_points)]

                offset_x = random.uniform(-5.0, 5.0)
                offset_y = random.uniform(-5.0, 5.0)
                location = carla.Location(
                    x=spawn_point.location.x + offset_x,
                    y=spawn_point.location.y + offset_y,
                    z=spawn_point.location.z
                )

                rotation = carla.Rotation(
                    pitch=spawn_point.rotation.pitch,
                    yaw=spawn_point.rotation.yaw + random.uniform(-15, 15),
                    roll=spawn_point.rotation.roll
                )

                transform = carla.Transform(location, rotation)

                vehicle = self.world.spawn_actor(vehicle_bp, transform)
                vehicle.set_autopilot(True)

                self.cooperative_vehicles.append(vehicle)
                spawned_vehicles.append(vehicle)

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
            ttl=5.0,
            priority=priority
        )

        self.v2x_messages.append(message)
        self.stats['total_messages'] += 1

        return message

    def broadcast_message(self, message: V2XMessage):
        """广播消息给范围内的车辆"""
        if message.sender_id not in self.vehicle_states:
            return []

        sender_state = self.vehicle_states[message.sender_id]
        sender_pos = sender_state.position

        recipients = []

        for vehicle in self.ego_vehicles + self.cooperative_vehicles:
            if vehicle.id != message.sender_id and vehicle.id in self.vehicle_states:
                receiver_state = self.vehicle_states[vehicle.id]
                receiver_pos = receiver_state.position

                distance = math.sqrt(
                    (sender_pos[0] - receiver_pos[0]) ** 2 +
                    (sender_pos[1] - receiver_pos[1]) ** 2 +
                    (sender_pos[2] - receiver_pos[2]) ** 2
                )

                if distance <= self.communication_range:
                    recipients.append(vehicle.id)

                    self.message_buffer[vehicle.id].append({
                        'message': message,
                        'receive_time': time.time(),
                        'signal_strength': 1.0 - (distance / self.communication_range)
                    })

        if recipients:
            self.stats['successful_transmissions'] += len(recipients)

            try:
                self._save_v2x_message(message, recipients)
            except Exception as e:
                print(f"保存V2X消息失败: {e}")

        return recipients

    def share_perception_data(self, vehicle_id: int, detected_objects: List[Dict]):
        """共享感知数据"""
        if not detected_objects:
            return None

        try:
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
                priority=2
            )

            recipients = self.broadcast_message(message)

            if recipients:
                self._fuse_shared_perception(vehicle_id, detected_objects, recipients)

            return message
        except Exception as e:
            print(f"共享感知数据失败: {e}")
            return None

    def share_traffic_warning(self, vehicle_id: int, warning_type: str,
                              location: Tuple[float, float, float],
                              severity: str = 'medium'):
        """共享交通警告"""
        try:
            warning_data = {
                'warning_type': warning_type,
                'location': location,
                'severity': severity,
                'timestamp': time.time(),
                'source_vehicle': vehicle_id
            }

            message = self.create_v2x_message(
                vehicle_id,
                'warning',
                warning_data,
                priority=3
            )

            self.broadcast_message(message)

            return message
        except Exception as e:
            print(f"共享交通警告失败: {e}")
            return None

    def share_pedestrian_warning(self, vehicle_id: int, pedestrian_location: Tuple[float, float, float],
                                 distance: float, speed: float, pedestrian_id: Optional[int] = None):
        """共享行人警告"""
        try:
            # 风险评估
            if distance < 5.0:
                severity = 'critical'
            elif distance < 10.0:
                severity = 'high'
            elif distance < 20.0:
                severity = 'medium'
            else:
                severity = 'low'

            warning_data = {
                'warning_type': 'pedestrian',
                'pedestrian_location': pedestrian_location,
                'distance': distance,
                'vehicle_speed': speed,
                'timestamp': time.time(),
                'source_vehicle': vehicle_id,
                'pedestrian_id': pedestrian_id,
                'severity': severity,
                'recommended_action': self._get_recommended_action(distance, speed, severity)
            }

            message = self.create_v2x_message(
                vehicle_id,
                'warning',
                warning_data,
                priority=4  # 行人警告优先级最高
            )

            recipients = self.broadcast_message(message)

            # 更新统计
            self.stats['pedestrian_warnings_sent'] += 1
            if severity in ['critical', 'high']:
                self.stats['safety_alerts'] += 1

            return message, recipients
        except Exception as e:
            print(f"共享行人警告失败: {e}")
            return None, []

    def _get_recommended_action(self, distance: float, speed: float, severity: str) -> str:
        """获取推荐的安全措施"""
        if severity == 'critical':
            return "立即紧急制动，准备避让"
        elif severity == 'high':
            return "减速至20km/h以下，准备制动"
        elif severity == 'medium':
            return "减速至30km/h，保持警惕"
        else:
            return "保持当前速度，注意观察"

    def _fuse_shared_perception(self, source_id: int, objects: List[Dict], recipients: List[int]):
        """融合共享的感知数据"""
        fused_objects = []

        for obj in objects:
            global_obj = obj.copy()
            global_obj['source_vehicles'] = [source_id]
            global_obj['confidence'] = obj.get('confidence', 0.8)

            matched = False
            for existing_obj in self.shared_objects:
                if self._objects_match(global_obj, existing_obj):
                    existing_obj['source_vehicles'].append(source_id)
                    existing_obj['confidence'] = min(1.0, existing_obj.get('confidence', 0) + 0.1)
                    existing_obj['update_time'] = time.time()
                    matched = True
                    break

            if not matched:
                global_obj['detection_time'] = time.time()
                fused_objects.append(global_obj)

        self.shared_objects.extend(fused_objects)

        current_time = time.time()
        self.shared_objects = [
            obj for obj in self.shared_objects
            if current_time - obj.get('detection_time', 0) < 10.0
        ]

        if fused_objects:
            self.stats['collaborative_detections'] += len(fused_objects)

    def _objects_match(self, obj1: Dict, obj2: Dict, distance_threshold: float = 5.0) -> bool:
        """判断两个对象是否匹配"""
        if obj1.get('class') != obj2.get('class'):
            return False

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
            shared_data.append(obj)

        return shared_data

    def coordinate_maneuvers(self, maneuvers: List[Dict]):
        """协调多车辆机动"""
        coordinated = []

        for i, maneuver in enumerate(maneuvers):
            coordinated_maneuver = maneuver.copy()
            coordinated_maneuver['sequence'] = i
            coordinated_maneuver['start_time'] = time.time() + i * 2.0
            coordinated.append(coordinated_maneuver)

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

        v2x_messages_dir = os.path.join(self.coop_dir, "v2x_messages")
        os.makedirs(v2x_messages_dir, exist_ok=True)

        filename = f"v2x_{int(time.time() * 1000)}_{message.sender_id}_{message.message_type}.json"
        filepath = os.path.join(v2x_messages_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(message_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"写入V2X消息文件失败: {e}")
            raise

    def save_shared_perception(self, frame_num: int):
        """保存共享感知数据"""
        try:
            data = {
                'frame_id': frame_num,
                'timestamp': time.time(),
                'shared_objects': self.shared_objects,
                'active_vehicles': len(self.ego_vehicles + self.cooperative_vehicles),
                'stats': self.stats
            }

            shared_perception_dir = os.path.join(self.coop_dir, "shared_perception")
            os.makedirs(shared_perception_dir, exist_ok=True)

            filepath = os.path.join(shared_perception_dir, f"frame_{frame_num:06d}.json")

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存共享感知数据失败: {e}")

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
                self.cooperative_vehicles)),
            'safety_metrics': {
                'pedestrian_warnings': self.stats['pedestrian_warnings_sent'],
                'safety_alerts': self.stats['safety_alerts'],
                'collaborative_detections': self.stats['collaborative_detections']
            }
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