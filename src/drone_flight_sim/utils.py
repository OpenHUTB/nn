# utils.py
"""工具函数模块

本模块提供通用的辅助函数，包括：
- 带倒计时的等待功能
- 位置信息格式化
- 分隔线打印
"""

# 导入 time 模块，用于实现 sleep 延时功能
import time


def wait_with_countdown(seconds: int, message: str = "等待"):
    """带倒计时的等待函数

    在控制台显示倒计时，并在等待期间打印进度信息。

    参数:
        seconds (int): 需要等待的秒数
        message (str): 等待时显示的提示消息，默认为"等待"
    """
    # 从指定秒数开始倒数到 1
    for i in range(seconds, 0, -1):
        # 打印等待信息，使用 \r 将光标回到行首实现覆盖效果
        # end="\r" 确保每次打印都覆盖上一行的内容
        print(f"   {message} {i} 秒...", end="\r")
        # 暂停 1 秒
        time.sleep(1)
    # 等待结束后清空该行（打印空格覆盖原有内容）
    print(" " * 50, end="\r")


def format_position(pos) -> str:
    """格式化无人机位置输出

    将位置对象格式化为易读的字符串格式。

    参数:
        pos: 包含 x_val, y_val, z_val 属性的位置对象

    返回:
        str: 格式化后的位置字符串，格式为 "(x, y, z)"
    """
    # 使用 f-string 格式化输出，保留一位小数
    return f"({pos.x_val:.1f}, {pos.y_val:.1f}, {pos.z_val:.1f})"


def print_separator(char: str = "=", length: int = 60):
    """打印分隔线

    在控制台打印一条由指定字符组成的分隔线，用于美化输出。

    参数:
        char (str): 分隔线使用的字符，默认为 "="
        length (int): 分隔线长度，默认为 60 个字符
    """
    # 使用字符重复打印分隔线
    print(char * length)
