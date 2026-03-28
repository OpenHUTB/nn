#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
结果可视化工具
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

class Visualizer:
    """结果可视化器"""
    
    def __init__(self, save_path="results/"):
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # 设置中文字体（避免警告）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_speed_profile(self, data, detections=None):
        """绘制速度曲线"""
        timestamps = [d['timestamp'] for d in data]
        speeds = [d['speed'] for d in data]
        speed_limits = [d['speed_limit'] for d in data]
        
        plt.figure(figsize=(12, 6))
        
        # 绘制速度曲线
        plt.plot(timestamps, speeds, 'b-', label='实际速度', linewidth=2)
        plt.plot(timestamps, speed_limits, 'r--', label='限速', linewidth=2)
        
        # 标记超速区域
        overspeed = [s > l for s, l in zip(speeds, speed_limits)]
        plt.fill_between(timestamps, speeds, speed_limits, 
                        where=overspeed, color='red', alpha=0.3, label='超速区域')
        
        # 标记红灯区域
        red_lights = [d['traffic_light'] == 'Red' for d in data]
        plt.fill_between(timestamps, 0, max(speeds) + 10,
                        where=red_lights, color='yellow', alpha=0.2, label='红灯时段')
        
        # 标记检测到的违规
        if detections and 'speeding' in detections:
            for v in detections['speeding'][:50]:  # 只标记前50个
                plt.scatter(v['timestamp'], v['speed'], 
                          color='red', s=30, zorder=5, marker='x')
        
        plt.xlabel('时间 (秒)', fontsize=12)
        plt.ylabel('速度 (km/h)', fontsize=12)
        plt.title('行驶速度曲线 (CARLA 仿真数据)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.save_path / 'speed_profile.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 速度曲线图已保存: {self.save_path}/speed_profile.png")
    
    def plot_violation_stats(self, violations, ground_truth=None):
        """绘制违规统计图"""
        violation_types = list(violations.keys())
        detected_counts = [len(violations[t]) for t in violation_types]
        
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(violation_types))
        width = 0.35
        
        # 检测结果
        bars1 = plt.bar(x - width/2, detected_counts, width, 
                       label='检测到的违规', color='#FF6B6B')
        
        # 真实标签（如果有）
        if ground_truth:
            actual_counts = [len(ground_truth.get(t, [])) for t in violation_types]
            bars2 = plt.bar(x + width/2, actual_counts, width,
                          label='真实违规', color='#4ECDC4', alpha=0.7)
        
        # 标签
        labels = {'speeding': '超速', 'red_light': '闯红灯', 'lane_violation': '压线'}
        plt.xticks(x, [labels.get(t, t) for t in violation_types])
        
        # 添加数值标签
        for bar in bars1:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(int(bar.get_height())), ha='center', va='bottom')
        
        if ground_truth:
            for bar in bars2:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(int(bar.get_height())), ha='center', va='bottom')
        
        plt.xlabel('违规类型', fontsize=12)
        plt.ylabel('次数', fontsize=12)
        plt.title('违规行为统计 (CARLA 仿真数据)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        plt.savefig(self.save_path / 'violation_stats.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 统计图已保存: {self.save_path}/violation_stats.png")
    
    def plot_trajectory(self, data):
        """绘制行驶轨迹"""
        x = [d['location_x'] for d in data]
        y = [d['location_y'] for d in data]
        speeds = [d['speed'] for d in data]
        
        plt.figure(figsize=(10, 8))
        
        scatter = plt.scatter(x, y, c=speeds, cmap='viridis', 
                            s=10, alpha=0.6)
        plt.colorbar(scatter, label='速度 (km/h)')
        
        plt.xlabel('X 位置 (米)', fontsize=12)
        plt.ylabel('Y 位置 (米)', fontsize=12)
        plt.title('行驶轨迹 (CARLA 仿真数据)', fontsize=14, fontweight='bold')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.save_path / 'trajectory.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 轨迹图已保存: {self.save_path}/trajectory.png")
    
    def save_report(self, data, violations, accuracy=None):
        """保存详细报告"""
        report_path = self.save_path / 'report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("CARLA 仿真数据违规驾驶行为检测报告\n")
            f.write("="*60 + "\n\n")
            
            # 数据来源说明
            f.write("【数据来源】\n")
            f.write("  数据来源: CARLA 自动驾驶仿真器\n")
            f.write("  数据类型: 驾驶场景仿真数据\n")
            f.write("  场景包含: 正常行驶、超速、闯红灯、压线\n\n")
            
            # 数据统计
            total_frames = len(data)
            total_time = data[-1]['timestamp']
            speeds = [d['speed'] for d in data]
            
            f.write("【数据统计】\n")
            f.write(f"  总帧数: {total_frames}\n")
            f.write(f"  总时长: {total_time:.1f} 秒\n")
            f.write(f"  平均速度: {np.mean(speeds):.1f} km/h\n")
            f.write(f"  最高速度: {max(speeds):.1f} km/h\n")
            f.write(f"  最低速度: {min(speeds):.1f} km/h\n\n")
            
            # 检测统计
            f.write("【违规检测统计】\n")
            total_violations = 0
            for vtype, records in violations.items():
                count = len(records)
                total_violations += count
                vtype_names = {'speeding': '超速', 'red_light': '闯红灯', 'lane_violation': '压线'}
                f.write(f"  {vtype_names.get(vtype, vtype)}: {count} 次\n")
            
            f.write(f"\n总违规次数: {total_violations}\n\n")
            
            # 准确率评估（如果有）
            if accuracy:
                f.write("【检测准确率】\n")
                for vtype, metrics in accuracy.items():
                    vtype_names = {'speeding': '超速', 'red_light': '闯红灯', 'lane_violation': '压线'}
                    f.write(f"  {vtype_names.get(vtype, vtype)}: ")
                    f.write(f"{metrics['accuracy']:.1f}% ")
                    f.write(f"({metrics['detected']}/{metrics['actual']})\n")
                f.write("\n")
            
            # 详细记录
            f.write("="*60 + "\n")
            f.write("【详细违规记录】\n")
            f.write("="*60 + "\n")
            
            for vtype, records in violations.items():
                if records:
                    vtype_names = {'speeding': '超速', 'red_light': '闯红灯', 'lane_violation': '压线'}
                    f.write(f"\n{vtype_names.get(vtype, vtype)}:\n")
                    for i, record in enumerate(records[:20]):  # 只显示前20条
                        f.write(f"  {i+1}. 时间: {record['timestamp']:.1f}s")
                        if 'speed' in record:
                            f.write(f", 速度: {record['speed']:.1f} km/h")
                        if 'limit' in record:
                            f.write(f", 限速: {record['limit']:.1f} km/h")
                        if 'lane_type' in record:
                            f.write(f", 车道: {record['lane_type']}")
                        f.write("\n")
                    
                    if len(records) > 20:
                        f.write(f"  ... 还有 {len(records)-20} 条记录\n")
        
        print(f"✅ 报告已保存: {report_path}")