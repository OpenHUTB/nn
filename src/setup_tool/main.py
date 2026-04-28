#!/usr/bin/env python3
"""
Setup Tool - Setup.bat Progress Display Demo

演示 setup.bat 的进度显示功能，无需实际执行安装操作。

Usage:
    python main.py
"""

import time
import sys
from datetime import datetime, timedelta


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


def main():
    """按顺序执行 setup 进度显示演示"""
    print()
    print("=" * 80)
    print("              Setup.bat Progress Display - DEMO VERSION")
    print("=" * 80)
    print("  This is a demonstration of the progress display features.")
    print("  No actual downloads or installations will be performed.")
    print("=" * 80)
    print()

    demo = ProgressDemo()

    # Step 1
    demo.show_progress("Checking hutb_downloader.exe")
    time.sleep(1)
    demo.step_result("skip", "hutb_downloader.exe already exists")

    # Step 2
    demo.show_progress("Checking dependencies directory")
    time.sleep(1)
    demo.step_result("skip", "dependencies directory already exists")

    # Step 3
    demo.show_progress("Checking for existing processes")
    time.sleep(1)
    demo.step_result("ok", "Killed 2 process(es) on port 2000")

    # Step 4
    demo.show_progress("Checking hutb directory")
    time.sleep(1)
    demo.step_result("skip", "hutb directory already exists")

    # Step 5
    demo.show_progress("Checking 7zip prerequisites")
    time.sleep(1)
    demo.step_result("ok", "7zip extracted successfully")

    # Step 6
    demo.show_progress("Checking miniconda3 prerequisites")
    time.sleep(1)
    print("  Extracting miniconda3 (simulated delay)...")
    time.sleep(2)
    demo.step_result("ok", "miniconda3 extracted successfully")

    # Step 7
    demo.show_progress("Validating Python environment")
    time.sleep(1)
    demo.step_result("ok", "Virtual environment found")
    demo.step_result("ok", "Python interpreter found")
    demo.step_result("ok", "numpy_tutorial.py found")

    # Step 8
    demo.show_progress("Initializing Python environment")
    time.sleep(1)
    print("  Activating virtual environment...")
    print("  Python version: 3.10.0")
    demo.step_result("ok", "Python environment ready")

    # Step 9
    demo.show_progress("Starting numpy_tutorial.py")
    time.sleep(1)
    demo.step_result("ok", "numpy_tutorial.py process started")

    # Step 10
    demo.show_progress("Waiting for service to be ready")
    time.sleep(2)
    print("  Waiting for service startup...")
    time.sleep(1)
    demo.step_result("ok", "numpy_tutorial.py is ready at http://192.168.1.100:3000")

    # Step 11
    demo.show_progress("Starting CarlaUE4.exe")
    time.sleep(1)
    demo.step_result("ok", "CarlaUE4.exe started")

    # Summary
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
