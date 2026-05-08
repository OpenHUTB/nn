import numpy as np


class PIDController:
    """6-DOF PID 控制器（x, y, z, yaw）"""

    def __init__(self, kp=1.0, ki=0.01, kd=0.5, max_vel=3.0):
        self.kp = np.array([kp] * 3)
        self.ki = np.array([ki] * 3)
        self.kd = np.array([kd] * 3)
        self.max_vel = max_vel

        self.prev_error = np.zeros(3)
        self.integral = np.zeros(3)
        self.prev_time = None

    def compute(self, current_pos, target_pos, dt=0.1):
        """计算速度命令

        Args:
            current_pos: (3,) 当前位置
            target_pos: (3,) 目标位置
            dt: 时间步长
        Returns:
            vx, vy, vz: 速度命令
        """
        error = target_pos - current_pos

        # 积分项（限幅防止积分饱和）
        self.integral += error * dt
        self.integral = np.clip(self.integral, -10, 10)

        # 微分项
        derivative = (error - self.prev_error) / dt if dt > 0 else np.zeros(3)
        self.prev_error = error

        # PID输出
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = np.clip(output, -self.max_vel, self.max_vel)

        return output[0], output[1], output[2]

    def reset(self):
        self.prev_error = np.zeros(3)
        self.integral = np.zeros(3)
