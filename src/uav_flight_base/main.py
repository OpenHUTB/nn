#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""uav_flight_base 模块统一入口。

课程要求：入口文件名为 main.*，所有功能支持 launch 启动。
本文件把 4 个任务收敛到一个命令：

    python3 main.py --list                 查看全部任务
    python3 main.py --task 1               键盘遥控（运动学仿真）
    python3 main.py --task 2 --mode lidar  激光雷达感知 + 反应式避障
    python3 main.py --task 2 --mode vision 相机颜色检测 + 视觉伺服
    python3 main.py --task 3               栅格建图 + A* 导航
    python3 main.py --task 4 --stage collect|train|drive   端到端网络

所有任务都在 Ubuntu 20.04 + ROS Noetic + Gazebo 11 上验证通过。
"""

import argparse
import os
import subprocess
import sys

PKG = "uav_flight_base"

TASKS = {
    1: {
        "name": "键盘遥控 + 运动学仿真",
        "desc": "键盘输入 → /uav/cmd_vel → 运动学模型 → /uav/odom + TF",
        "launch": "main.launch",
        "args": {"mode": "mock"},
    },
    2: {
        "name": "传感器感知 + 运动控制",
        "desc": "lidar: 360 线激光 → 扇区最近距离 → 反应式避障；"
                "vision: HSV 颜色检测 → PID 视觉伺服",
        "launch": None,   # 按 mode 选择
        "args": {},
    },
    3: {
        "name": "建图 + 导航",
        "desc": "激光 ray casting 占据栅格 → 8 邻域 A* → 纯追踪飞行",
        "launch": "nav.launch",
        "args": {},
    },
    4: {
        "name": "端到端模型（图像 → 控制）",
        "desc": "专家采数据 → 训练网络 → 网络实时飞行",
        "launch": None,
        "args": {},
    },
}

LAUNCH_2 = {"lidar": "avoid.launch", "vision": "servo.launch"}


def show_list():
    print("=" * 66)
    print(" uav_flight_base —— 无人机仿真模块（任务 1~4）")
    print("=" * 66)
    for k in sorted(TASKS):
        t = TASKS[k]
        print(" [任务 %d] %s" % (k, t["name"]))
        print("          %s" % t["desc"])
        if t["launch"]:
            print("          启动: roslaunch %s %s" % (PKG, t["launch"]))
        elif k == 2:
            print("          启动: roslaunch %s avoid.launch  |  servo.launch" % PKG)
        else:
            print("          启动: stage=collect / train / drive（见 README）")
        print("")
    print(" 用法示例： python3 main.py --task 3")
    print("=" * 66)


def run_launch(launch, extra, dry):
    cmd = ["roslaunch", PKG, launch] + extra
    print("[main] 执行: " + " ".join(cmd))
    if dry:
        print("[main] --dry-run：只打印命令，不启动")
        return 0
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        sys.stderr.write("[main] 找不到 roslaunch，请先 source /opt/ros/noetic/setup.bash\n")
        return 127


def run_task4(stage, dry):
    home = os.path.expanduser("~")
    data = os.path.join(home, "bc_data")
    scripts = os.path.join(home, "catkin_ws/src", PKG, "scripts")

    if stage == "collect":
        cmd = ["rosrun", PKG, "uav_expert_collect.py"]
    elif stage == "train":
        cmd = [sys.executable, os.path.join(scripts, "uav_train_np.py"), data, "300"]
    elif stage == "drive":
        cmd = ["rosrun", PKG, "uav_np_driver.py", os.path.join(data, "weights.npz")]
    else:
        sys.stderr.write("[main] --stage 只能是 collect / train / drive\n")
        return 2

    print("[main] 执行: " + " ".join(cmd))
    if dry:
        print("[main] --dry-run：只打印命令，不启动")
        return 0
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser(description="无人机仿真模块统一入口")
    ap.add_argument("--task", type=int, choices=[1, 2, 3, 4], help="要运行的任务编号")
    ap.add_argument("--mode", default="lidar", choices=["lidar", "vision"],
                    help="任务2 的感知方式")
    ap.add_argument("--stage", default="collect", choices=["collect", "train", "drive"],
                    help="任务4 的阶段")
    ap.add_argument("--gui", action="store_true", help="显示 Gazebo 界面")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    ap.add_argument("--list", action="store_true", help="列出全部任务")
    args, extra = ap.parse_known_args()

    if args.list or args.task is None:
        show_list()
        return 0

    gui = "true" if args.gui else "false"

    if args.task == 1:
        # main.launch 没有 gui 参数，只有 teleop / csv
        return run_launch(TASKS[1]["launch"],
                          ["mode:=mock"] + extra, args.dry_run)

    if args.task == 2:
        launch = LAUNCH_2[args.mode]
        return run_launch(launch, ["gui:=%s" % gui] + extra, args.dry_run)

    if args.task == 3:
        return run_launch(TASKS[3]["launch"], ["gui:=%s" % gui] + extra, args.dry_run)

    return run_task4(args.stage, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
