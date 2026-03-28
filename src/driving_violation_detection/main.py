#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
驾驶违规行为检测系统 - 主程序
使用真实的 CARLA 仿真数据
"""

import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from data_loader import CarlaDataLoader
from detector import ViolationDetector
from visualizer import Visualizer


def main():
    """主函数"""
    print("\n" + "="*60)
    print("驾驶违规行为检测系统")
    print("基于 CARLA 自动驾驶仿真器真实数据")
    print("="*60)
    
    # 1. 初始化数据加载器
    print("\n📂 初始化 CARLA 数据加载器...")
    loader = CarlaDataLoader()
    
    # 显示可用的数据集
    datasets = loader.list_available_datasets()
    if datasets:
        print("\n📁 可用的 CARLA 数据集:")
        for ds in datasets:
            print(f"   - {ds['town']} / {ds['weather']}")
    
    # 2. 加载数据（如果数据不存在，会提示运行采集程序）
    print("\n🔍 正在加载 CARLA 仿真数据...")
    data = loader.load_data(town='Town01', weather='ClearNoon')
    
    if data is None:
        print("\n⚠️ 未找到 CARLA 数据文件")
        print("\n请先运行数据采集程序:")
        print("  python data/download_real_carla_data.py")
        print("\n该程序将生成基于真实 CARLA 仿真的驾驶数据")
        return
    
    # 3. 显示数据摘要
    summary = loader.get_data_summary()
    print("\n📊 CARLA 数据摘要:")
    print(f"   数据来源: {summary['source']}")
    print(f"   城镇: {summary.get('town', 'Unknown')}")
    print(f"   天气: {summary.get('weather', 'Unknown')}")
    print(f"   总帧数: {summary['total_frames']}")
    print(f"   时长: {summary['duration']:.1f} 秒")
    print(f"   平均速度: {summary['avg_speed']:.1f} km/h")
    print(f"   速度范围: {summary['min_speed']:.1f} - {summary['max_speed']:.1f} km/h")
    
    if summary.get('violation_stats'):
        print(f"\n   违规场景分布:")
        for vtype, count in summary['violation_stats'].items():
            vtype_names = {'speeding': '超速', 'red_light': '闯红灯', 'lane': '压线'}
            print(f"     {vtype_names.get(vtype, vtype)}: {count} 帧")
    
    # 4. 初始化检测器
    print("\n🔍 初始化违规检测器...")
    detector = ViolationDetector()
    
    # 5. 处理所有帧
    print("\n🚗 开始违规检测...")
    violation_count = 0
    print_count = 0
    first_violations = []
    
    for frame in data:
        violations = detector.process_frame(frame)
        violation_count += len(violations)
        
        # 记录前15个违规
        if print_count < 15:
            for v in violations:
                print_count += 1
                if v['type'] == 'speeding':
                    print(f"  ⚠️ 超速违规: {v['speed']:.1f} km/h (限速: {v['limit']:.1f} km/h)")
                elif v['type'] == 'red_light':
                    print(f"  🚦 闯红灯违规: 红灯时行驶 (速度: {v['speed']:.1f} km/h)")
                elif v['type'] == 'lane_violation':
                    print(f"  📏 压线违规: 行驶在{v['lane_type']}")
    
    if violation_count > 15:
        print(f"  ... 还有 {violation_count - 15} 条违规记录")
    
    # 6. 显示统计结果
    detector.print_summary()
    
    # 7. 获取真实标签并评估准确率
    ground_truth = loader.get_ground_truth()
    accuracy = detector.evaluate_accuracy(ground_truth)
    
    if any(accuracy):
        print("\n📊 检测准确率评估:")
        vtype_names = {'speeding': '超速', 'red_light': '闯红灯', 'lane_violation': '压线'}
        for vtype, metrics in accuracy.items():
            if metrics['actual'] > 0:
                print(f"   {vtype_names.get(vtype, vtype)}: {metrics['accuracy']:.1f}% "
                      f"({metrics['detected']}/{metrics['actual']})")
            elif metrics['detected'] > 0:
                print(f"   {vtype_names.get(vtype, vtype)}: 误报 ({metrics['detected']}次)")
    
    # 8. 生成可视化图表
    print("\n📈 生成可视化图表...")
    visualizer = Visualizer()
    visualizer.plot_speed_profile(data, detector.violations)
    visualizer.plot_violation_stats(detector.violations, ground_truth)
    visualizer.plot_trajectory(data)
    
    # 9. 保存详细报告
    visualizer.save_report(data, detector.violations, accuracy)
    
    print("\n✅ 检测完成！")
    print("📁 结果保存在 results/ 目录")
    print("\n" + "="*60)
    print("项目说明:")
    print("  ✅ 数据来源: CARLA 自动驾驶仿真器")
    print("  ✅ 数据版本: CARLA 0.9.13")
    print("  ✅ 包含场景: 正常驾驶、超速、闯红灯、压线")
    print("  ✅ 检测算法: 基于规则的违规检测")
    print("  ✅ 评估指标: 准确率、召回率")
    print("="*60)


if __name__ == "__main__":
    main()