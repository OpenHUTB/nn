# -*- coding: utf-8 -*-
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