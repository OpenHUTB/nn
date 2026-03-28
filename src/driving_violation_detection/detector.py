#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
违规驾驶行为检测器
"""

import numpy as np

class ViolationDetector:
    """违规行为检测器"""
    
    def __init__(self):
        # 检测阈值
        self.speed_margin = 5  # 超速阈值（km/h）
        self.red_light_speed = 0.5  # 闯红灯判定速度
        
        # 违规记录
        self.violations = {
            'speeding': [],
            'red_light': [],
            'lane_violation': []
        }
        
        # 状态跟踪
        self.red_light_start = None
    
    def detect_speeding(self, speed, speed_limit, timestamp):
        """检测超速"""
        if speed > speed_limit + self.speed_margin:
            return {
                'type': 'speeding',
                'timestamp': timestamp,
                'speed': speed,
                'limit': speed_limit,
                'exceed': speed - speed_limit
            }
        return None
    
    def detect_red_light(self, traffic_light, speed, timestamp):
        """检测闯红灯"""
        if traffic_light == 'Red':
            if self.red_light_start is None:
                self.red_light_start = timestamp
            elif speed > self.red_light_speed:
                return {
                    'type': 'red_light',
                    'timestamp': timestamp,
                    'speed': speed,
                    'duration': timestamp - self.red_light_start
                }
        else:
            self.red_light_start = None
        return None
    
    def detect_lane_violation(self, lane_type, timestamp):
        """检测压线违规"""
        forbidden_lanes = ['Shoulder', 'Sidewalk', 'BikeLane']
        
        if lane_type in forbidden_lanes:
            return {
                'type': 'lane_violation',
                'timestamp': timestamp,
                'lane_type': lane_type
            }
        return None
    
    def process_frame(self, frame_data):
        """处理单帧数据"""
        violations = []
        
        # 提取数据
        timestamp = frame_data['timestamp']
        speed = frame_data['speed']
        speed_limit = frame_data['speed_limit']
        traffic_light = frame_data['traffic_light']
        lane_type = frame_data['lane_type']
        
        # 超速检测
        speeding = self.detect_speeding(speed, speed_limit, timestamp)
        if speeding:
            violations.append(speeding)
            self.violations['speeding'].append(speeding)
        
        # 闯红灯检测
        red_light = self.detect_red_light(traffic_light, speed, timestamp)
        if red_light:
            violations.append(red_light)
            self.violations['red_light'].append(red_light)
        
        # 压线检测
        lane = self.detect_lane_violation(lane_type, timestamp)
        if lane:
            violations.append(lane)
            self.violations['lane_violation'].append(lane)
        
        return violations
    
    def get_statistics(self):
        """获取统计信息"""
        stats = {}
        for vtype, records in self.violations.items():
            stats[vtype] = len(records)
        return stats
    
    def evaluate_accuracy(self, ground_truth):
        """评估检测准确率（与真实标签对比）"""
        if not ground_truth:
            return {}
        
        results = {}
        for vtype in self.violations.keys():
            detected = len(self.violations[vtype])
            actual = len(ground_truth.get(vtype, []))
            
            if actual > 0:
                accuracy = detected / actual * 100
                results[vtype] = {
                    'detected': detected,
                    'actual': actual,
                    'accuracy': accuracy
                }
            else:
                results[vtype] = {
                    'detected': detected,
                    'actual': actual,
                    'accuracy': 0
                }
        
        return results
    
    def print_summary(self):
        """打印摘要"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("检测结果统计:")
        print("="*60)
        print(f"  超速: {stats['speeding']} 次")
        print(f"  闯红灯: {stats['red_light']} 次")
        print(f"  压线: {stats['lane_violation']} 次")
        print("="*60)