#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
think_twice 主入口
符合 nn 仓库规范：src/think_twice/main.py
功能：支持训练、评测、数据采集
"""

import sys
import os

def show_help():
    print("="*50)
    print("think_twice 主程序入口")
    print("="*50)
    print("使用方法：")
    print("  python main.py train        # 启动神经网络训练")
    print("  python main.py eval         # 启动 CARLA 闭环评测")
    print("  python main.py collect      # 启动数据采集")
    print("="*50)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        show_help()
        sys.exit(1)

    cmd = sys.argv[1]
    base = os.path.dirname(os.path.abspath(__file__))

    if cmd == "train":
        os.chdir(os.path.join(base, "open_loop_training"))
        os.system("python main_train.py --config configs/thinktwice.py")
    elif cmd == "eval":
        os.chdir(os.path.join(base, "leaderboard", "leaderboard"))
        os.system("python main_eval.py")
    elif cmd == "collect":
        print("数据采集功能即将开放")
    else:
        show_help()
