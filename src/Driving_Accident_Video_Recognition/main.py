"""
主程序：驾驶事故视频识别工具
"""
import sys
import os
import argparse

# 确保当前目录可被搜索
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 直接导入同目录的文件（彻底避免模块包问题）
from config import REQUIRED_PACKAGES, PYPI_MIRROR, PERSON_VEHICLE_DISTANCE, ACCIDENT_CONTINUOUS_FRAMES
from dependencies import install_dependencies  # 直接导入同目录的dependencies.py
from detector import AccidentDetector


def parse_args():
    parser = argparse.ArgumentParser(description="驾驶事故视频识别")
    parser.add_argument("--source", "-s", default=0, help="检测源：0=摄像头/视频路径")
    parser.add_argument("--language", "-l", default="zh", choices=["zh", "en"], help="标注语言")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        print("🚀 启动驾驶事故检测...")
        # 安装依赖
        install_dependencies(REQUIRED_PACKAGES, PYPI_MIRROR)
        # 启动检测
        detector = AccidentDetector()
        detector.run_detection(language=args.language)
    except KeyboardInterrupt:
        print("\n🛑 程序中断")
    finally:
        print("👋 程序退出")


if __name__ == "__main__":
    main()
