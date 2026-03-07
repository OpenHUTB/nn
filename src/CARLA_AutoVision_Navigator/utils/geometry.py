# -*- coding: utf-8 -*-
"""
Project: CARLA AutoVision Navigator
Module: Utility - Geometric Computation
Version: v1.0.0
Description: 几何计算工具库。提供向量空间下的车速换算、航向角偏差计算等自动驾驶核心数学函数。
Author: wangadsa
License: MIT License
"""
import math
import numpy as np


def get_speed(vehicle):
    """计算车辆当前速度 (km/h)"""
    v = vehicle.get_velocity()
    return 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def get_steer_angle(vehicle, waypoint):
    """
    计算主车与目标导航点之间的转向偏差角度
    """
    v_transform = vehicle.get_transform()
    v_location = v_transform.location
    v_yaw = v_transform.rotation.yaw

    # 获取车辆前进方向向量 (基于 Yaw 角)
    v_vec = np.array([math.cos(math.radians(v_yaw)), math.sin(math.radians(v_yaw)), 0.0])

    # 获取目标导航点向量
    w_vec = np.array([waypoint.transform.location.x - v_location.x,
                      waypoint.transform.location.y - v_location.y, 0.0])

    # 计算两个向量之间的余弦夹角
    dot = np.dot(v_vec, w_vec)
    mag1 = np.linalg.norm(v_vec)
    mag2 = np.linalg.norm(w_vec)

    if mag1 * mag2 == 0: return 0

    cos_angle = dot / (mag1 * mag2)
    angle = math.acos(max(-1.0, min(1.0, cos_angle)))

    # 利用叉乘判断左右方向
    cross = v_vec[0] * w_vec[1] - v_vec[1] * w_vec[0]
    if cross < 0:
        angle *= -1

    return angle