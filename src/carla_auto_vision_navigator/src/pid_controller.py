<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Project: CARLA AutoVision Navigator
Module: Control - PID Algorithm
Version: v1.0.0
Description: 车辆运动控制器。实现标准的比例-积分-微分（PID）算法，用于主车的纵向速度补偿与横向转向修正。
Author: wangadsa
License: MIT License
"""
import collections


class PIDController:
    """
    基础 PID 控制器，用于车辆的纵向速度控制
    """

    def __init__(self, k_p, k_i, k_d, dt=0.05):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self._dt = dt
        self._error_buffer = collections.deque(maxlen=10)

    def run_step(self, target_speed, current_speed):
        """
        执行一个计算步长，返回控制值
        :param target_speed: 目标速度 (km/h)
        :param current_speed: 当前速度 (km/h)
        :return: 控制信号 [-1.0, 1.0] (正数为油门，负数为刹车)
        """
        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        # PID 公式计算
        output = (self.k_p * error) + (self.k_d * _de) + (self.k_i * _ie)

        # 限制输出范围在 [-1.0, 1.0]
        return max(-1.0, min(1.0, output))
=======
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PIDController:
    def __init__(self, kp=1.0, ki=0.1, kd=0.05):
        self.kp = kp  # 比例系数
        self.ki = ki  # 积分系数
        self.kd = kd  # 微分系数

        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def compute(self, target_value, current_value):
        """计算PID输出"""
        current_time = time.time()
        dt = current_time - self.last_time if current_time - self.last_time > 0 else 0.001

        # 计算误差
        error = target_value - current_value

        # 比例项
        proportional = self.kp * error

        # 积分项（防积分饱和）
        self.integral += error * dt
        self.integral = max(min(self.integral, 10.0), -10.0)
        integral = self.ki * self.integral

        # 微分项
        derivative = self.kd * (error - self.last_error) / dt

        # 总输出
        output = proportional + integral + derivative

        # 更新状态
        self.last_error = error
        self.last_time = current_time

        return output

    def control_vehicle(self, vehicle, action):
        """控制车辆执行决策动作"""
        # 获取车辆当前状态
        current_speed = vehicle.get_velocity()
        current_speed_kmh = 3.6 * (current_speed.x ** 2 + current_speed.y ** 2 + current_speed.z ** 2) ** 0.5

        # 速度控制（PID）
        speed_output = self.compute(action["speed"], current_speed_kmh)

        # 构造车辆控制指令
        control = carla.VehicleControl()
        control.throttle = max(min(speed_output / 100, 1.0), 0.0) if speed_output > 0 else 0.0
        control.brake = max(min(-speed_output / 100, 1.0), 0.0) if speed_output < 0 else action["brake"]
        control.steer = action["steer"]
        control.hand_brake = True if action["speed"] == 0 else False

        # 执行控制
        vehicle.apply_control(control)
        logger.info(f"当前速度：{current_speed_kmh:.1f}km/h | 目标速度：{action['speed']}km/h | 转向：{action['steer']}")
>>>>>>> upstream/main
