"""
安全监控模块 - Safety Monitor
实时监控ACC系统安全性，记录安全事件
"""

import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from collections import deque


@dataclass
class SafetyEvent:
    """安全事件记录"""
    timestamp: float
    event_type: str  # 'warning', 'danger', 'emergency', 'recovery'
    distance: float
    ego_speed: float
    target_speed: float
    TTC: float
    message: str


class SafetyMonitor:
    """
    安全监控器
    监控和记录ACC系统的安全状态
    """

    def __init__(self, max_events: int = 100):
        """
        初始化安全监控器

        Args:
            max_events: 最大记录事件数
        """
        self.max_events = max_events
        self.events: deque = deque(maxlen=max_events)
        self.start_time = time.time()
        self.total_distance_violations = 0
        self.total_speed_violations = 0
        self.total_ttc_warnings = 0
        self.emergency_count = 0

    def record_event(self,
                    event_type: str,
                    distance: float,
                    ego_speed: float,
                    target_speed: float,
                    TTC: float,
                    message: str):
        """
        记录安全事件

        Args:
            event_type: 事件类型
            distance: 距离
            ego_speed: 自车速度
            target_speed: 目标速度
            TTC: 碰撞时间
            message: 事件消息
        """
        event = SafetyEvent(
            timestamp=time.time() - self.start_time,
            event_type=event_type,
            distance=distance,
            ego_speed=ego_speed,
            target_speed=target_speed,
            TTC=TTC,
            message=message
        )
        self.events.append(event)

        # 更新统计
        if event_type == 'distance_violation':
            self.total_distance_violations += 1
        elif event_type == 'speed_violation':
            self.total_speed_violations += 1
        elif event_type == 'ttc_warning':
            self.total_ttc_warnings += 1
        elif event_type == 'emergency':
            self.emergency_count += 1

    def get_recent_events(self, n: int = 10) -> List[SafetyEvent]:
        """
        获取最近的事件

        Args:
            n: 事件数量

        Returns:
            最近的事件列表
        """
        return list(self.events)[-n:]

    def get_statistics(self) -> Dict[str, any]:
        """
        获取安全统计信息

        Returns:
            统计信息字典
        """
        total_events = len(self.events)
        if total_events == 0:
            return {
                'total_events': 0,
                'distance_violations': 0,
                'speed_violations': 0,
                'ttc_warnings': 0,
                'emergency_count': 0,
                'safety_score': 100.0
            }

        # 计算安全评分
        safety_score = 100.0
        safety_score -= self.emergency_count * 10
        safety_score -= self.total_distance_violations * 2
        safety_score -= self.total_speed_violations * 1
        safety_score -= self.total_ttc_warnings * 0.5
        safety_score = max(0.0, min(100.0, safety_score))

        return {
            'total_events': total_events,
            'distance_violations': self.total_distance_violations,
            'speed_violations': self.total_speed_violations,
            'ttc_warnings': self.total_ttc_warnings,
            'emergency_count': self.emergency_count,
            'safety_score': safety_score
        }

    def print_summary(self):
        """打印安全摘要"""
        stats = self.get_statistics()

        print("\n" + "=" * 60)
        print("安全监控摘要")
        print("=" * 60)
        print(f"总事件数: {stats['total_events']}")
        print(f"距离违规: {stats['distance_violations']}")
        print(f"速度违规: {stats['speed_violations']}")
        print(f"TTC警告: {stats['ttc_warnings']}")
        print(f"紧急情况: {stats['emergency_count']}")
        print(f"安全评分: {stats['safety_score']:.1f}/100")
        print("=" * 60)

        # 打印最近的事件
        recent = self.get_recent_events(5)
        if recent:
            print("\n最近事件:")
            for event in recent:
                print(f"[{event.timestamp:.1f}s] {event.event_type}: {event.message}")

    def reset(self):
        """重置监控数据"""
        self.events.clear()
        self.start_time = time.time()
        self.total_distance_violations = 0
        self.total_speed_violations = 0
        self.total_ttc_warnings = 0
        self.emergency_count = 0
