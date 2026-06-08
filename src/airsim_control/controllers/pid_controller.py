import numpy as np
import time


class PIDController:
    """
    6-DOF PID Controller
    控制:
        x, y, z, yaw
    """

    def __init__(
            self,
            kp=1.2,
            ki=0.02,
            kd=0.4,
            max_vel=3.0,
            max_yaw_rate=1.0,
            integral_limit=5.0,
            alpha=0.7
    ):

        # PID参数
        self.kp = np.array([kp, kp, kp])
        self.ki = np.array([ki, ki, ki])
        self.kd = np.array([kd, kd, kd])

        # 输出限制
        self.max_vel = max_vel
        self.max_yaw_rate = max_yaw_rate

        # 积分限制（防积分饱和）
        self.integral_limit = integral_limit

        # 低通滤波系数
        self.alpha = alpha

        # 状态量
        self.prev_error = np.zeros(3)
        self.integral = np.zeros(3)
        self.prev_output = np.zeros(3)

        self.prev_time = time.time()

    def compute(
            self,
            current_pos,
            target_pos,
            current_yaw=0.0,
            target_yaw=0.0
    ):
        """
        计算PID控制输出

        Args:
            current_pos: 当前坐标 [x, y, z]
            target_pos: 目标坐标 [x, y, z]
            current_yaw: 当前航向角
            target_yaw: 目标航向角

        Returns:
            vx, vy, vz, yaw_rate
        """

        # 自动计算dt
        current_time = time.time()
        dt = current_time - self.prev_time

        # 防止dt异常
        if dt <= 0 or dt > 1:
            dt = 0.1

        self.prev_time = current_time

        # 位置误差
        error = np.array(target_pos) - np.array(current_pos)

        # =========================
        # 积分项（积分分离）
        # =========================
        small_error_mask = np.abs(error) < 5.0
        self.integral[small_error_mask] += error[small_error_mask] * dt

        # 积分限幅
        self.integral = np.clip(
            self.integral,
            -self.integral_limit,
            self.integral_limit
        )

        # =========================
        # 微分项
        # =========================
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        # =========================
        # PID输出
        # =========================
        output = (
                self.kp * error +
                self.ki * self.integral +
                self.kd * derivative
        )

        # =========================
        # 输出平滑（低通滤波）
        # =========================
        output = (
                self.alpha * self.prev_output +
                (1 - self.alpha) * output
        )

        self.prev_output = output

        # =========================
        # 限制最大速度
        # =========================
        output = np.clip(
            output,
            -self.max_vel,
            self.max_vel
        )

        # =========================
        # Yaw控制
        # =========================
        yaw_error = target_yaw - current_yaw

        # 角度归一化 [-pi, pi]
        yaw_error = np.arctan2(
            np.sin(yaw_error),
            np.cos(yaw_error)
        )

        yaw_rate = np.clip(
            1.0 * yaw_error,
            -self.max_yaw_rate,
            self.max_yaw_rate
        )

        return (
            float(output[0]),
            float(output[1]),
            float(output[2]),
            float(yaw_rate)
        )

    def reset(self):
        """重置控制器状态"""

        self.prev_error = np.zeros(3)
        self.integral = np.zeros(3)
        self.prev_output = np.zeros(3)

        self.prev_time = time.time()
