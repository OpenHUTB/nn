#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO11 车辆计数系统 - 主入口
功能：启动车辆检测与计数，支持多版本切换
优化：代码规范、结构清晰、注释统一
"""

"""
YOLO11n Vehicle Counter - 主入口文件
=====================================

这个脚本是YOLO11n车辆计数项目的入口点，提供了命令行接口来运行车辆检测和视频处理。
支持运行原始版本和改进版本的车辆计数脚本。

使用方法:
    python main.py [--model MODEL_PATH] [--input INPUT_VIDEO] [--output OUTPUT_VIDEO] [--version VERSION]

参数:
    --model      模型文件路径 (默认: models/yolo11n.pt)
    --input      输入视频路径 (默认: dataset/sample.mp4)
    --output     输出视频路径 (默认: res/sample_res.mp4)
    --version    脚本版本: original(原始版本) 或 improved(改进版本) (默认: original)
    --help       显示帮助信息
"""

import sys
import os
import argparse

# 添加scripts目录到Python路径，以便可以导入其中的模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

def main():
    """
    主启动函数
    解析命令行参数 → 检查文件路径 → 加载对应版本 → 执行车辆计数
    """
   
    parser = argparse.ArgumentParser(
        description='YOLO11n Vehicle Counter - 车辆检测与计数系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    # 使用默认路径运行(原始版本)
    python main.py

    # 运行改进版本
    python main.py --version improved

    # 指定自定义路径(原始版本)
    python main.py --model models/custom_model.pt --input videos/test.mp4 --output results/output.mp4

    # 指定自定义路径(改进版本)
    python main.py --version improved --model models/custom_model.pt --input videos/test.mp4 --output res/improved/output.mp4

    # 使用相对路径
    python main.py --input ../videos/cars.mp4 --output ../results/cars_counted.mp4
        """
    )

    parser.add_argument(
        '--model',
        default='models/yolo11n.pt',
        help='YOLO模型文件路径 (默认: models/yolo11n.pt)'
    )

    parser.add_argument(
        '--input',
        default='dataset/sample.mp4',
        help='输入视频文件路径 (默认: dataset/sample.mp4)'
    )

    parser.add_argument(
        '--output',
        default='res/sample_res.mp4',
        help='输出视频文件路径 (默认: res/sample_res.mp4)'
    )

    parser.add_argument(
        '--ground-truth',
        default=None,
        help='ground truth文件路径，用于精度衡量 (可选)'
    )

    parser.add_argument(
        '--version',
        choices=['original', 'improved', 'updown', 'carla', 'region'],
        default='original',
        help='脚本版本: original(原始版本), improved(改进版本), updown(上下行计数), carla(CARLA视频专用) 或 region(区域计数) (默认: original)'
    )

    args = parser.parse_args()

    # 打印运行信息
    print("=" * 60)
    print("YOLO11n Vehicle Counter - 启动")
    version_name = '原始版本' if args.version == 'original' else ('改进版本' if args.version == 'improved' else '上下行计数版本')
    print(f"版本: {args.version} ({version_name})")
    print("=" * 60)
    print(f"📁 模型路径: {args.model}")
    print(f"🎬 输入视频: {args.input}")
    print(f"💾 输出视频: {args.output}")
    print("=" * 60)

    # 检查文件是否存在
    if not os.path.exists(args.model):
        print(f"❌ 错误: 模型文件不存在: {args.model}")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"❌ 错误: 输入视频不存在: {args.input}")
        sys.exit(1)

    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 创建输出目录: {output_dir}")

    # 运行车辆计数脚本
    try:
        if args.version == 'original':
            # 运行原始版本
            from yolo_vehicle_counter import main as run_counter
            print("🚀 正在运行原始版本...")
        elif args.version == 'updown':
            # 运行上下行计数版本
            from yolo_vehicle_counter_updown import main as run_counter
            print("🚀 正在运行上下行计数版本...")
        elif args.version == 'carla':
            # 运行CARLA视频专用版本
            from yolo_vehicle_counter_carla import main as run_counter
            print("🚀 正在运行CARLA视频专用版本...")
        elif args.version == 'region':
            # 运行区域计数版本
            from scripts.yolo_vehicle_counter_region import main as run_counter
            print("🚀 正在运行区域计数版本...")
        else:
            # 运行改进版本
            from yolo_vehicle_counter_improved import main as run_counter
            print("🚀 正在运行改进版本...")

        # 传入参数到脚本
        run_counter(args.model, args.input, args.output, args.ground_truth)
        print("✅ 车辆计数完成！")

    except ImportError as e:
        print(f"❌ 错误: 无法导入模块 - {e}")
        if args.version == 'original':
            print("确保scripts目录中有yolo_vehicle_counter.py文件")
        elif args.version == 'updown':
            print("确保scripts目录中有yolo_vehicle_counter_updown.py文件")
        elif args.version == 'carla':
            print("确保scripts目录中有yolo_vehicle_counter_carla.py文件")
        else:
            print("确保scripts目录中有yolo_vehicle_counter_improved.py文件")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()