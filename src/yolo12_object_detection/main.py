#!/usr/bin/env python3
"""
YOLO12 目标检测项目入口文件

提供训练、验证和推理的统一入口
"""

import os
import sys
import argparse
import warnings

# 添加 scripts 目录到路径
scripts_dir = os.path.join(os.path.dirname(__file__), 'scripts')
sys.path.insert(0, scripts_dir)

warnings.filterwarnings('ignore')


def train():
    """训练入口"""
    print("🚀 启动训练...")
    from train import main as train_main
    train_main()


def val():
    """验证入口"""
    print("📊 启动验证...")
    from val import main as val_main
    val_main()


def predict(source=None, model_path=None):
    """推理入口"""
    print("🔍 启动推理...")
    from ultralytics import YOLO

    if model_path is None:
        model_path = 'runs/train/baseline/weights/best.pt'

    model = YOLO(model_path)
    results = model.predict(
        source=source,
        imgsz=640,
        conf=0.25,
        iou=0.45,
        project='runs/detect',
        name='predict',
        save=True,
        show=False
    )

    return results


def main():
    parser = argparse.ArgumentParser(description='YOLO12 目标检测项目')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.set_defaults(func=train)

    # 验证命令
    val_parser = subparsers.add_parser('val', help='验证模型')
    val_parser.set_defaults(func=val)

    # 推理命令
    predict_parser = subparsers.add_parser('predict', help='推理检测')
    predict_parser.add_argument('--source', type=str, default=None, help='输入源 (图片/视频/摄像头)')
    predict_parser.add_argument('--model', type=str, default=None, help='模型路径')
    predict_parser.set_defaults(func=lambda args: predict(args.source, args.model))

    # 帮助信息
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # 执行对应命令
    if args.command == 'predict':
        args.func(args)
    else:
        args.func()


if __name__ == '__main__':
    main()