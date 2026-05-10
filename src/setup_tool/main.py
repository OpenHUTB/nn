#!/usr/bin/env python3
"""
Setup Tool - Setup.bat Progress Display Demo

演示 setup.bat 的进度显示功能，无需实际执行安装操作。

Usage:
    python main.py
"""

import argparse
import time
import sys


class ProgressDemo:
    """用于演示 setup 过程中的进度显示、状态输出和汇总信息"""

    def __init__(self, total_steps=11):
        self.total_steps = total_steps
        self.current_step = 0
        self.skipped = 0
        self.downloaded = 0
        self.errors = 0
        self.start_time = time.time()
        self.step_times = []

    def show_progress(self, description):
        """更新当前步骤并打印带百分比和 ETA 的进度信息"""
        self.current_step += 1
        # 根据当前步骤计算进度百分比
        percent = int(self.current_step * 100 / self.total_steps)

        # 计算已用时间
        elapsed = int(time.time() - self.start_time)
        elapsed_m, elapsed_s = divmod(elapsed, 60)

        # 估算剩余时间
        if self.current_step > 1:
            # 基于已执行步骤的平均耗时估算剩余时间
            avg_time = elapsed / (self.current_step - 1)
            remaining = avg_time * (self.total_steps - self.current_step + 1)
            remain_m, remain_s = divmod(int(remaining), 60)
        else:
            remain_m, remain_s = "--", "--"

        # 构建进度条
        bar_size = percent // 2
        # 生成固定宽度的文本进度条
        progress_bar = "#" * bar_size + " " * (50 - bar_size)

        # 显示进度
        print()
        print("=" * 80)
        print(f"  [{progress_bar}] {percent}%   Step {self.current_step}/{self.total_steps}")
        print(f"  Current: {description}")
        print(f"  Elapsed: {elapsed_m}m {elapsed_s:02d}s  |  ETA: {remain_m}m {remain_s}s")
        print("=" * 80)
        print()

        sys.stdout.flush()

    def step_result(self, status, message):
        """显示单个步骤的执行结果，并统计不同状态的数量"""
        if status == "skip":
            self.skipped += 1
            print(f"  [SKIP] {message}")
        elif status == "download":
            self.downloaded += 1
            print(f"  [DOWNLOAD] {message}")
        elif status == "error":
            self.errors += 1
            print(f"  [ERROR] {message}")
        else:
            print(f"  [OK] {message}")
        print()
        sys.stdout.flush()

    def show_summary(self):
        """输出整个演示流程的最终汇总信息"""
        # 统计总耗时并输出最终汇总
        total_time = int(time.time() - self.start_time)
        tm, ts = divmod(total_time, 60)

        print()
        print("=" * 80)
        print("                         SETUP SUMMARY")
        print("=" * 80)
        print(f"  Total Steps:        {self.current_step}/{self.total_steps}")
        print(f"  Total Time:         {tm}m {ts:02d}s")
        print(f"  Skipped:            {self.skipped}")
        print(f"  Downloaded:         {self.downloaded}")
        print(f"  Errors:             {self.errors}")
        print("=" * 80)
        print()

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Setup tool progress display demo"
    )
    parser.add_argument(
        "--speed",
        choices=["fast", "normal", "slow"],
        default="normal",
        help="Set demo speed: fast, normal, or slow",
    )
    parser.add_argument(
        "--mode",
        choices=["basic", "full"],
        default="full",
        help="Set demo mode: basic or full",
    )
    return parser.parse_args()


def main():
   """按顺序执行 setup 进度显示演示。"""
   args = parse_args()

   speed_map = {
        "fast": 0.3,
        "normal": 1.0,
        "slow": 1.5,
    }
   speed_factor = speed_map[args.speed]

   print()
   print("=" * 80)
   print("              Setup.bat Progress Display - DEMO VERSION")
   print("=" * 80)
   print("  This is a demonstration of the progress display features.")
   print("  No actual downloads or installations will be performed.")
   print(f"  Demo speed: {args.speed}")
   print(f"  Demo mode: {args.mode}")
   print("=" * 80)
   print()


   full_steps = [
        {
            "description": "Checking hutb_downloader.exe",
            "actions": [
                {"type": "sleep", "seconds": 1},
                {"type": "result", "status": "skip", "message": "hutb_downloader.exe already exists"},
            ],
        },
        {
            "description": "Checking dependencies directory",
            "actions": [
                {"type": "sleep", "seconds": 1},
                {"type": "result", "status": "skip", "message": "dependencies directory already exists"},
            ],
        },
        {
            "description": "Checking for existing processes",
            "actions": [
                {"type": "sleep", "seconds": 1},
                {"type": "result", "status": "ok", "message": "Killed 2 process(es) on port 2000"},
            ],
        },
        {
            "description": "Checking hutb directory",
            "actions": [
                {"type": "sleep", "seconds": 1},
                {"type": "result", "status": "skip", "message": "hutb directory already exists"},
            ],
        },
        {
            "description": "Checking 7zip prerequisites",
            "actions": [
                {"type": "sleep", "seconds": 1},
                {"type": "result", "status": "ok", "message": "7zip extracted successfully"},
            ],
        },
        {
            "description": "Checking miniconda3 prerequisites",
            "actions": [
                {"type": "sleep", "seconds": 1},
                {"type": "print", "message": "  Extracting miniconda3 (simulated delay)..."},
                {"type": "sleep", "seconds": 2},
                {"type": "result", "status": "ok", "message": "miniconda3 extracted successfully"},
            ],
        },
        {
            "description": "Validating Python environment",
            "actions": [
                {"type": "sleep", "seconds": 1},
                {"type": "result", "status": "ok", "message": "Virtual environment found"},
                {"type": "result", "status": "ok", "message": "Python interpreter found"},
                {"type": "result", "status": "ok", "message": "numpy_tutorial.py found"},
            ],
        },
        {
            "description": "Initializing Python environment",
            "actions": [
                {"type": "sleep", "seconds": 1},
                {"type": "print", "message": "  Activating virtual environment..."},
                {"type": "print", "message": "  Python version: 3.10.0"},
                {"type": "result", "status": "ok", "message": "Python environment ready"},
            ],
        },
        {
            "description": "Starting numpy_tutorial.py",
            "actions": [
                {"type": "sleep", "seconds": 1},
                {"type": "result", "status": "ok", "message": "numpy_tutorial.py process started"},
            ],
        },
        {
            "description": "Waiting for service to be ready",
            "actions": [
                {"type": "sleep", "seconds": 2},
                {"type": "print", "message": "  Waiting for service startup..."},
                {"type": "sleep", "seconds": 1},
                {"type": "result", "status": "ok", "message": "numpy_tutorial.py is ready at http://192.168.1.100:3000"},
            ],
        },
        {
            "description": "Starting CarlaUE4.exe",
            "actions": [
                {"type": "sleep", "seconds": 1},
                {"type": "result", "status": "ok", "message": "CarlaUE4.exe started"},
            ],
        },
    ]
   
   basic_steps = [
        full_steps[0],
        full_steps[2],
        full_steps[6],
        full_steps[8],
        full_steps[10],
    ]
   
   if args.mode == "basic":
        steps = basic_steps
   else:
        steps = full_steps

   demo = ProgressDemo(total_steps=len(steps))

    # 使用统一的数据结构描述步骤，减少 main() 中的重复逻辑
   for step in steps:
        demo.show_progress(step["description"])

        for action in step["actions"]:
            if action["type"] == "sleep":
                time.sleep(action["seconds"] * speed_factor)
            elif action["type"] == "print":
                print(action["message"])
            elif action["type"] == "result":
                demo.step_result(action["status"], action["message"])

   demo.show_summary()

   print()
   print("=" * 80)
   print("                         DEMO COMPLETE")
   print("=" * 80)
   print()
   print("The progress display features demonstrated:")
   print("  - Real-time progress bar with percentage")
   print("  - Step numbering (1/11, 2/11, etc.)")
   print("  - Elapsed time tracking")
   print("  - Estimated time remaining (ETA)")
   print("  - Standardized status output ([OK], [SKIP], [DOWNLOAD], [ERROR])")
   print("  - Final summary with statistics")
   print()
   print("To use the actual setup script, run: setup.bat")
   print("=" * 80)
   print()


if __name__ == "__main__":
    main()
