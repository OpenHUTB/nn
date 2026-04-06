"""
物理引擎模块
"""
import numpy as np


class PhysicsEngine:
    """简单的无人机物理引擎"""

    def __init__(self, mass=1.0, gravity=9.81):
        """
        初始化物理引擎

        Args:
            mass: 无人机质量 (kg)
            gravity: 重力加速度 (m/s²)
        """
        self.mass = mass
        self.gravity = gravity

        # 物理参数
        self.drag_coefficient = 0.1  # 空气阻力系数
        self.max_thrust = 20.0  # 最大推力 (N)
        self.motor_response_time = 0.1  # 电机响应时间 (s)

        # 状态变量
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0, 0.0])
        self.orientation = np.array([0.0, 0.0, 0.0])  # [roll, pitch, yaw]
        self.angular_velocity = np.array([0.0, 0.0, 0.0])

        # 控制输入
        self.thrust = np.array([0.0, 0.0, self.mass * self.gravity])  # 初始推力平衡重力
        self.target_orientation = np.array([0.0, 0.0, 0.0])

        # 环境参数
        self.wind_velocity = np.array([0.0, 0.0, 0.0])
        self.air_density = 1.225  # 空气密度 kg/m³

        # 限制
        self.max_tilt_angle = np.radians(30)  # 最大倾斜角度
        self.max_altitude = 50.0  # 最大高度
        self.min_altitude = 0.0  # 最小高度

    def update(self, dt, control_input):
        """
        更新物理状态

        Args:
            dt: 时间步长 (秒)
            control_input: 控制输入字典
                {
                    'throttle': 油门 [0, 1],
                    'roll': 横滚角 (弧度),
                    'pitch': 俯仰角 (弧度),
                    'yaw_rate': 偏航率 (弧度/秒)
                }
        """
        # 更新控制输入
        self._update_control_input(control_input, dt)

        # 计算合力
        total_force = self._calculate_total_force()

        # 计算加速度 (F = ma)
        self.acceleration = total_force / self.mass

        # 更新速度 (v = v0 + a*t)
        self.velocity += self.acceleration * dt

        # 更新位置 (x = x0 + v*t)
        self.position += self.velocity * dt

        # 更新姿态
        self._update_orientation(dt)

        # 应用限制
        self._apply_constraints()

        return self._get_state()

    def _update_control_input(self, control_input, dt):
        """更新控制输入"""
        # 油门控制垂直推力
        throttle = control_input.get('throttle', 0.5)
        vertical_thrust = throttle * self.max_thrust

        # 目标姿态
        self.target_orientation[0] = control_input.get('roll', 0.0)
        self.target_orientation[1] = control_input.get('pitch', 0.0)

        # 限制最大倾斜角度
        self.target_orientation[0] = np.clip(
            self.target_orientation[0],
            -self.max_tilt_angle,
            self.max_tilt_angle
        )
        self.target_orientation[1] = np.clip(
            self.target_orientation[1],
            -self.max_tilt_angle,
            self.max_tilt_angle
        )

        # 偏航控制
        yaw_rate = control_input.get('yaw_rate', 0.0)
        self.target_orientation[2] += yaw_rate * dt

        # 计算推力向量
        # 根据姿态计算推力方向
        roll, pitch, yaw = self.target_orientation

        # 旋转矩阵 (ZYX顺序)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        rotation_matrix = Rz @ Ry @ Rx

        # 推力向量 (初始向上)
        thrust_vector_local = np.array([0.0, 0.0, vertical_thrust])

        # 旋转到世界坐标系
        self.thrust = rotation_matrix @ thrust_vector_local

    def _calculate_total_force(self):
        """计算总合力"""
        # 重力
        gravity_force = np.array([0.0, -self.mass * self.gravity, 0.0])

        # 推力
        thrust_force = self.thrust

        # 空气阻力 (与速度方向相反)
        relative_velocity = self.velocity - self.wind_velocity
        speed = np.linalg.norm(relative_velocity)
        if speed > 0:
            drag_direction = -relative_velocity / speed
            drag_magnitude = 0.5 * self.drag_coefficient * self.air_density * speed ** 2
            drag_force = drag_magnitude * drag_direction
        else:
            drag_force = np.array([0.0, 0.0, 0.0])

        # 总合力
        total_force = gravity_force + thrust_force + drag_force

        return total_force

    def _update_orientation(self, dt):
        """更新姿态"""
        # 简单的姿态控制 (PD控制器)
        kp = 5.0  # 比例增益
        kd = 0.5  # 微分增益

        # 计算姿态误差
        orientation_error = self.target_orientation - self.orientation

        # 计算角加速度
        angular_acceleration = kp * orientation_error - kd * self.angular_velocity

        # 更新角速度
        self.angular_velocity += angular_acceleration * dt

        # 更新姿态
        self.orientation += self.angular_velocity * dt

        # 限制角速度
        max_angular_velocity = np.radians(180)  # 最大180度/秒
        self.angular_velocity = np.clip(
            self.angular_velocity,
            -max_angular_velocity,
            max_angular_velocity
        )

    def _apply_constraints(self):
        """应用物理约束"""
        # 高度限制
        if self.position[1] > self.max_altitude:
            self.position[1] = self.max_altitude
            self.velocity[1] = min(self.velocity[1], 0)

        if self.position[1] < self.min_altitude:
            self.position[1] = self.min_altitude
            self.velocity[1] = max(self.velocity[1], 0)

        # 地面碰撞检测
        if self.position[1] <= self.min_altitude + 0.01:
            self.velocity = np.array([0.0, 0.0, 0.0])
            self.acceleration = np.array([0.0, 0.0, 0.0])

    def _get_state(self):
        """获取当前状态"""
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'acceleration': self.acceleration.copy(),
            'orientation': self.orientation.copy(),
            'angular_velocity': self.angular_velocity.copy(),
            'thrust': self.thrust.copy()
        }

    def reset(self, position=None, orientation=None):
        """重置物理引擎状态"""
        if position is not None:
            self.position = np.array(position)
        else:
            self.position = np.array([0.0, 0.0, 0.0])

        if orientation is not None:
            self.orientation = np.array(orientation)
        else:
            self.orientation = np.array([0.0, 0.0, 0.0])

        self.velocity = np.array([0.0, 0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0, 0.0])
        self.angular_velocity = np.array([0.0, 0.0, 0.0])
        self.thrust = np.array([0.0, 0.0, self.mass * self.gravity])