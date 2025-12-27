"""模式控制和速度限制模块"""
import numpy as np

# 模式常量
MODE_WALK = 0  # 行走模式
MODE_RUN = 1   # 奔跑模式
MODE_DISTURBANCE = 2  # 抗扰动测试模式

# 模式名称
mode_names = ["行走", "奔跑", "抗扰动测试"]

# 速度限制（根据模式动态调整）
# [x前进最大速度, x后退最大速度, y最大速度, yaw最大速度]
cmd_max_walk = np.array([0.8, 0.8, 0.65, 1.3])  # 行走模式
cmd_max_run = np.array([2.0, 0.8, 0.8, 1.3])  # 奔跑模式
cmd_max_disturbance = np.array([0.8, 0.8, 0.65, 1.3])  # 抗扰动模式（允许控制速度，使用行走模式的速度限制）


def get_cmd_max_for_mode(mode):
    """根据模式获取速度限制"""
    if mode == MODE_WALK:
        return cmd_max_walk
    elif mode == MODE_RUN:
        return cmd_max_run
    elif mode == MODE_DISTURBANCE:
        return cmd_max_disturbance
    else:
        return cmd_max_walk  # 默认

