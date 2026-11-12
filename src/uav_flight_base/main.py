#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""uav_flight_base 模块入口。

用法:
    python3 main.py                 # 默认 mock 模式（不需要任何模拟器）
    python3 main.py --mode gazebo   # 接 Gazebo（需要额外安装 uav 模型）
    python3 main.py --list          # 只打印可用节点，不启动

约定（对齐 OpenHUTB 仓库规范）:
    · 入口文件名为 main.*
    · ROS 启动配置为 launch/main.launch
    · 所有参数通过 roslaunch args 或 rosparam 传入，不写死在代码里
"""

import argparse
import os
import subprocess
import sys

MODES = ("mock", "gazebo")


def build_launch_cmd(mode, extra_args):
    return ["roslaunch", "uav_flight_base", "main.launch",
            "mode:={}".format(mode)] + extra_args


def main():
    ap = argparse.ArgumentParser(description="无人机基础飞行模块（键盘遥控 / 状态监控）")
    ap.add_argument("--mode", default="mock", choices=MODES,
                    help="mock=纯软件仿真，gazebo=接 Gazebo 世界")
    ap.add_argument("--list", action="store_true", help="打印节点清单后退出")
    ap.add_argument("--no-launch", action="store_true",
                    help="不调用 roslaunch，只打印将要执行的命令")
    args, extra = ap.parse_known_args()

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    scripts = sorted(
        f for f in os.listdir(os.path.join(pkg_dir, "scripts"))
        if f.endswith(".py")
    ) if os.path.isdir(os.path.join(pkg_dir, "scripts")) else []

    if args.list:
        print("package : uav_flight_base")
        print("mode    : {}".format(args.mode))
        print("nodes   :")
        for s in scripts:
            print("  - {}".format(s))
        return 0

    cmd = build_launch_cmd(args.mode, extra)
    print("[uav_flight_base] mode = {}".format(args.mode))
    print("[uav_flight_base] exec: {}".format(" ".join(cmd)))

    if args.no_launch:
        return 0

    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        sys.stderr.write("找不到 roslaunch，请先 source /opt/ros/noetic/setup.bash\n")
        return 127


if __name__ == "__main__":
    sys.exit(main())
