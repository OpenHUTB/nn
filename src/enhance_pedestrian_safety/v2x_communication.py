import socket
import threading
import json
import time
import struct
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import queue


@dataclass
class V2XMessage:
    """V2X通信消息"""
    message_id: str
    sender_id: str
    message_type: str  # 'bsm', 'spat', 'map', 'rsm', 'rsa'
    data: Dict[str, Any]
    timestamp: float
    ttl: float
    priority: int
    position: Optional[tuple] = None


class V2XCommunication:
    """V2X通信模拟器"""

    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('enabled', True)

        # 通信参数
        self.communication_range = config.get('communication_range', 300.0)  # 通信范围（米）
        self.bandwidth = config.get('bandwidth', 10.0)  # 带宽（Mbps）
        self.latency_mean = config.get('latency_mean', 0.05)  # 平均延迟（秒）
        self.latency_std = config.get('latency_std', 0.01)  # 延迟标准差
        self.packet_loss_rate = config.get('packet_loss_rate', 0.01)  # 丢包率

        # 消息管理
        self.message_queue = queue.PriorityQueue()
        self.received_messages = []
        self.message_counter = 0

        # 网络模拟
        self.network_nodes = {}  # 节点ID -> 节点信息
        self.connections = {}  # 连接状态

        # 统计信息
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'messages_dropped': 0,
            'total_latency': 0.0,
            'bandwidth_used': 0.0
        }

        # 启动消息处理线程
        if self.enabled:
            self.running = True
            self.processor_thread = threading.Thread(target=self._message_processor, daemon=True)
            self.processor_thread.start()

    def register_node(self, node_id: str, position: tuple, capabilities: Dict):
        """注册网络节点（车辆）"""
        self.network_nodes[node_id] = {
            'position': position,
            'capabilities': capabilities,
            'last_seen': time.time(),
            'status': 'online'
        }

    def unregister_node(self, node_id: str):
        """注销网络节点"""
        if node_id in self.network_nodes:
            del self.network_nodes[node_id]

    def send_message(self, message: V2XMessage) -> bool:
        """发送V2X消息"""
        if not self.enabled:
            return False

        # 模拟网络延迟和丢包
        if np.random.random() < self.packet_loss_rate:
            self.stats['messages_dropped'] += 1
            return False

        # 添加延迟
        latency = np.random.normal(self.latency_mean, self.latency_std)
        delivery_time = time.time() + max(0, latency)

        # 添加到消息队列（按优先级和发送时间排序）
        priority = -message.priority  # 负号因为PriorityQueue是小顶堆
        self.message_queue.put((priority, delivery_time, message))

        self.stats['messages_sent'] += 1
        self.stats['total_latency'] += latency

        return True

    def broadcast_basic_safety_message(self, node_id: str, vehicle_data: Dict) -> Optional[V2XMessage]:
        """广播基本安全消息（BSM）"""
        bsm_data = {
            'msgCnt': self.message_counter % 128,
            'id': node_id,
            'secMark': int(time.time() * 1000) % 60000,
            'position': vehicle_data.get('position', (0, 0, 0)),
            'accuracy': vehicle_data.get('accuracy', {'semiMajor': 1.0, 'semiMinor': 1.0}),
            'transmissionAndSpeed': {
                'transmission': vehicle_data.get('transmission', 'automatic'),
                'speed': vehicle_data.get('speed', 0.0)
            },
            'heading': vehicle_data.get('heading', 0.0),
            'steeringWheelAngle': vehicle_data.get('steering_wheel_angle', 0.0),
            'accelerationSet4Way': vehicle_data.get('acceleration', {'long': 0, 'lat': 0, 'vert': 0, 'yaw': 0}),
            'brakeSystemStatus': vehicle_data.get('brake_status', {'wheelBrakes': '00000'}),
            'vehicleSize': vehicle_data.get('size', {'width': 1.8, 'length': 4.5})
        }

        message = V2XMessage(
            message_id=f"bsm_{node_id}_{self.message_counter}",
            sender_id=node_id,
            message_type='bsm',
            data=bsm_data,
            timestamp=time.time(),
            ttl=1.0,
            priority=1,
            position=vehicle_data.get('position')
        )

        self.message_counter += 1

        if self.send_message(message):
            return message

        return None

    def broadcast_signal_phase_and_timing(self, intersection_id: str, spat_data: Dict) -> Optional[V2XMessage]:
        """广播信号相位和时序消息（SPaT）"""
        message = V2XMessage(
            message_id=f"spat_{intersection_id}_{self.message_counter}",
            sender_id=intersection_id,
            message_type='spat',
            data=spat_data,
            timestamp=time.time(),
            ttl=5.0,
            priority=2,
            position=spat_data.get('intersection_position')
        )

        self.message_counter += 1

        if self.send_message(message):
            return message

        return None

    def broadcast_map_data(self, map_id: str, map_data: Dict) -> Optional[V2XMessage]:
        """广播地图数据（MAP）"""
        message = V2XMessage(
            message_id=f"map_{map_id}_{self.message_counter}",
            sender_id=map_id,
            message_type='map',
            data=map_data,
            timestamp=time.time(),
            ttl=30.0,  # 地图数据TTL较长
            priority=1,
            position=map_data.get('reference_point')
        )

        self.message_counter += 1

        if self.send_message(message):
            return message

        return None

    def broadcast_roadside_safety_message(self, rsu_id: str, warning_data: Dict) -> Optional[V2XMessage]:
        """广播路侧安全消息（RSM/RSA）"""
        message = V2XMessage(
            message_id=f"rsm_{rsu_id}_{self.message_counter}",
            sender_id=rsu_id,
            message_type='rsm',
            data=warning_data,
            timestamp=time.time(),
            ttl=3.0,
            priority=3,  # 安全消息优先级高
            position=warning_data.get('event_position')
        )

        self.message_counter += 1

        if self.send_message(message):
            return message

        return None

    def get_reachable_nodes(self, sender_id: str, sender_position: tuple) -> List[str]:
        """获取可达的网络节点"""
        reachable = []

        for node_id, node_info in self.network_nodes.items():
            if node_id == sender_id:
                continue

            # 计算距离
            distance = self._calculate_distance(sender_position, node_info['position'])

            if distance <= self.communication_range:
                # 模拟信号衰减
                signal_strength = 1.0 - (distance / self.communication_range)
                if signal_strength > 0.3:  # 最小信号强度阈值
                    reachable.append(node_id)

        return reachable

    def _calculate_distance(self, pos1: tuple, pos2: tuple) -> float:
        """计算两点间距离"""
        if not pos1 or not pos2:
            return float('inf')

        return np.sqrt(
            (pos1[0] - pos2[0]) ** 2 +
            (pos1[1] - pos2[1]) ** 2 +
            (pos1[2] - pos2[2]) ** 2
        )

    def _message_processor(self):
        """消息处理线程"""
        while self.running:
            try:
                # 获取下一个要传递的消息
                if not self.message_queue.empty():
                    priority, delivery_time, message = self.message_queue.get_nowait()

                    current_time = time.time()

                    if current_time >= delivery_time:
                        # 消息已到传递时间
                        self._deliver_message(message)
                    else:
                        # 重新放回队列
                        self.message_queue.put((priority, delivery_time, message))
                        time.sleep(0.001)  # 短暂休眠
                else:
                    time.sleep(0.01)

            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                print(f"消息处理器错误: {e}")

    def _deliver_message(self, message: V2XMessage):
        """传递消息到接收者"""
        if not message.position:
            return

        # 获取可达节点
        reachable_nodes = self.get_reachable_nodes(message.sender_id, message.position)

        # 模拟带宽限制
        message_size = len(json.dumps(asdict(message)).encode('utf-8')) / 1024 / 1024  # MB
        bandwidth_required = message_size * 8 * len(reachable_nodes)  # Mbps

        if bandwidth_required > self.bandwidth * 0.8:  # 带宽超过80%时随机丢包
            drop_probability = min(1.0, bandwidth_required / self.bandwidth - 0.8)
            reachable_nodes = [n for n in reachable_nodes if np.random.random() > drop_probability]

        # 添加到接收消息列表
        for node_id in reachable_nodes:
            received_msg = {
                'message': asdict(message),
                'receiver_id': node_id,
                'receive_time': time.time(),
                'signal_strength': 1.0 - (self._calculate_distance(message.position, self.network_nodes[node_id][
                    'position']) / self.communication_range)
            }

            self.received_messages.append(received_msg)
            self.stats['messages_received'] += 1

        # 更新带宽使用统计
        self.stats['bandwidth_used'] += bandwidth_required

    def get_messages_for_node(self, node_id: str, message_types: List[str] = None) -> List[Dict]:
        """获取指定节点的消息"""
        messages = []

        for msg_record in self.received_messages:
            if msg_record['receiver_id'] == node_id:
                message = msg_record['message']
                if not message_types or message['message_type'] in message_types:
                    messages.append(msg_record)

        # 清理已处理的消息
        self.received_messages = [
            msg for msg in self.received_messages
            if msg['receiver_id'] != node_id or
               (message_types and msg['message']['message_type'] not in message_types)
        ]

        return messages

    def get_network_status(self) -> Dict:
        """获取网络状态"""
        return {
            'active_nodes': len(self.network_nodes),
            'stats': self.stats,
            'bandwidth_utilization': (self.stats['bandwidth_used'] / (
                        self.bandwidth * time.time())) * 100 if time.time() > 0 else 0,
            'message_delivery_rate': self.stats['messages_received'] / max(1, self.stats['messages_sent']),
            'average_latency': self.stats['total_latency'] / max(1, self.stats['messages_sent'])
        }

    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'messages_dropped': 0,
            'total_latency': 0.0,
            'bandwidth_used': 0.0
        }

    def stop(self):
        """停止通信模拟"""
        self.running = False
        if hasattr(self, 'processor_thread'):
            self.processor_thread.join(timeout=1.0)