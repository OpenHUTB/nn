# utils.py
"""工具函数"""

import time

def wait_with_countdown(seconds: int, message: str = "等待"):
    """带倒计时的等待"""
    for i in range(seconds, 0, -1):
        print(f"   {message} {i} 秒...", end="\r")
        time.sleep(1)
    print(" " * 50, end="\r")

def format_position(pos) -> str:
    """格式化位置输出"""
    return f"({pos.x_val:.1f}, {pos.y_val:.1f}, {pos.z_val:.1f})"

def print_separator(char: str = "=", length: int = 60):
    """打印分隔线"""
    print(char * length)