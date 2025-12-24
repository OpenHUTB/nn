"""
手势稳定性模块
负责手势的稳定性分析和优化
作者: xiaoshiyuan888
"""

import numpy as np
from collections import deque
import time
import statistics


class GestureStabilizer:
    """手势稳定性分析器"""

    def __init__(self, window_size=10):
        self.window_size = window_size
        self.gesture_history = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)
        self.timestamp_history = deque(maxlen=window_size)
        self.position_history = deque(maxlen=window_size)

        # 稳定性参数
        self.stability_score = 0.0
        self.last_stable_gesture = None
        self.last_stable_time = 0
        self.transition_count = 0

    def add_gesture(self, gesture, confidence, position=None):
        """添加手势到历史"""
        current_time = time.time()

        self.gesture_history.append(gesture)
        self.confidence_history.append(confidence)
        self.timestamp_history.append(current_time)

        if position is not None:
            self.position_history.append(position)

        # 检查手势转换
        if len(self.gesture_history) > 1:
            if self.gesture_history[-1] != self.gesture_history[-2]:
                self.transition_count += 1

        return self.analyze_stability()

    def analyze_stability(self):
        """分析手势稳定性"""
        if len(self.gesture_history) < 2:
            return {
                'stable': False,
                'stability_score': 0.0,
                'dominant_gesture': None,
                'confidence': 0.0
            }

        # 计算手势一致性
        gesture_counts = {}
        for gesture in self.gesture_history:
            if gesture not in gesture_counts:
                gesture_counts[gesture] = 0
            gesture_counts[gesture] += 1

        # 找出主导手势
        dominant_gesture = max(gesture_counts.items(), key=lambda x: x[1])[0]
        consistency_score = gesture_counts[dominant_gesture] / len(self.gesture_history)

        # 计算置信度稳定性
        if len(self.confidence_history) > 0:
            avg_confidence = np.mean(list(self.confidence_history))
            confidence_std = np.std(list(self.confidence_history)) if len(self.confidence_history) >= 2 else 0
            confidence_score = max(0, 1.0 - confidence_std)
        else:
            avg_confidence = 0
            confidence_score = 0

        # 计算时间稳定性
        if len(self.timestamp_history) >= 2:
            time_intervals = []
            for i in range(1, len(self.timestamp_history)):
                interval = self.timestamp_history[i] - self.timestamp_history[i - 1]
                time_intervals.append(interval)

            if len(time_intervals) > 0:
                avg_interval = np.mean(time_intervals)
                interval_std = np.std(time_intervals) if len(time_intervals) >= 2 else 0
                time_score = max(0, 1.0 - min(1.0, interval_std / 0.5))  # 0.5秒为标准间隔
            else:
                time_score = 1.0
        else:
            time_score = 1.0

        # 计算位置稳定性
        position_score = 1.0
        if len(self.position_history) >= 2:
            position_changes = []
            for i in range(1, len(self.position_history)):
                pos1 = self.position_history[i - 1]
                pos2 = self.position_history[i]
                if pos1 is not None and pos2 is not None:
                    distance = np.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)
                    position_changes.append(distance)

            if position_changes:
                avg_change = np.mean(position_changes)
                position_score = max(0, 1.0 - min(1.0, avg_change * 10))  # 缩放因子

        # 计算综合稳定性分数
        weights = {
            'consistency': 0.4,
            'confidence': 0.3,
            'time': 0.2,
            'position': 0.1
        }

        self.stability_score = (
                consistency_score * weights['consistency'] +
                confidence_score * weights['confidence'] +
                time_score * weights['time'] +
                position_score * weights['position']
        )

        # 检查是否稳定
        is_stable = self.stability_score > 0.7 and consistency_score > 0.6

        if is_stable and dominant_gesture != self.last_stable_gesture:
            self.last_stable_gesture = dominant_gesture
            self.last_stable_time = time.time()

        return {
            'stable': is_stable,
            'stability_score': self.stability_score,
            'dominant_gesture': dominant_gesture if is_stable else None,
            'confidence': avg_confidence,
            'consistency': consistency_score,
            'transition_count': self.transition_count
        }

    def get_stability_level(self):
        """获取稳定性等级"""
        if self.stability_score >= 0.8:
            return "excellent"
        elif self.stability_score >= 0.6:
            return "good"
        elif self.stability_score >= 0.4:
            return "fair"
        else:
            return "poor"

    def reset(self):
        """重置稳定性分析"""
        self.gesture_history.clear()
        self.confidence_history.clear()
        self.timestamp_history.clear()
        self.position_history.clear()
        self.stability_score = 0.0
        self.last_stable_gesture = None
        self.transition_count = 0

    def get_stats(self):
        """获取统计信息"""
        return {
            'history_size': len(self.gesture_history),
            'stability_score': self.stability_score,
            'stability_level': self.get_stability_level(),
            'last_stable_gesture': self.last_stable_gesture,
            'transition_count': self.transition_count,
            'avg_confidence': np.mean(list(self.confidence_history)) if self.confidence_history else 0,
            'gesture_variety': len(set(self.gesture_history))
        }