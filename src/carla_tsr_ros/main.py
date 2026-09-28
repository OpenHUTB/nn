#!/usr/bin/env python3
"""carla_tsr_ros 入口脚本

通过 subprocess 调用 roslaunch，统一入口。
"""
import os
import sys
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(description='Carla Traffic Sign Recognition ROS launcher')
    parser.add_argument('--image-dir', default=None, help='图片目录（默认包内 data 目录）')
    parser.add_argument('--model-path', default=None, help='YOLO 权重路径')
    parser.add_argument('--rate', type=float, default=2.0, help='图像发布频率（Hz）')
    args = parser.parse_args()

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    launch_file = os.path.join(pkg_dir, 'main.launch')

    cmd = ['roslaunch', launch_file]
    if args.image_dir:
        cmd.append(f'image_dir:={args.image_dir}')
    if args.model_path:
        cmd.append(f'model_path:={args.model_path}')
    cmd.append(f'publish_rate:={args.rate}')

    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("ERROR: roslaunch not found. Please run:")
        print("  source /opt/ros/noetic/setup.bash")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"roslaunch failed with code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == '__main__':
    main()
