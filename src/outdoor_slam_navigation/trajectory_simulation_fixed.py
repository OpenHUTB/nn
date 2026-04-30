# examples/trajectory_simulation_fixed.py
"""
轨迹模拟演示 - 单文件版本
"""

import math
import random
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple


import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 使用中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# ==================== 基础工具函数 ====================
def normalize_angle(angle: float) -> float:
    """将角度归一化到[-π, π]区间"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


# ==================== 机器人运动学类 ====================
class WheelParameters:
    def __init__(self, radius=0.1, separation=0.5, max_speed=1.0,
                 max_acceleration=0.5, wheel_base=0.3):
        self.radius = radius
        self.separation = separation
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.wheel_base = wheel_base

    @property
    def max_angular_speed(self):
        return (2 * self.max_speed) / self.separation

    def __str__(self):
        return f"WheelParameters(radius={self.radius}, separation={self.separation})"


class DifferentialDriveKinematics:
    def __init__(self, params=None):
        self.params = params or WheelParameters()
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.total_distance = 0.0
        self.total_rotation = 0.0
        self._trajectory = []
        self._recording = False
        self._time = 0.0

    @property
    def pose(self):
        return self.x, self.y, self.theta

    def set_pose(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = normalize_angle(theta)
        self._time = 0.0

    def move_with_velocity(self, linear_speed, angular_speed, dt=0.1):
        # 限制速度
        linear_speed = max(min(linear_speed, self.params.max_speed), -self.params.max_speed)
        max_angular = self.params.max_angular_speed
        angular_speed = max(min(angular_speed, max_angular), -max_angular)

        # 计算位移
        dx = linear_speed * math.cos(self.theta) * dt
        dy = linear_speed * math.sin(self.theta) * dt
        dtheta = angular_speed * dt

        # 更新位姿
        self.x += dx
        self.y += dy
        self.theta = normalize_angle(self.theta + dtheta)
        self._time += dt

        # 更新累计值
        self.total_distance += math.sqrt(dx ** 2 + dy ** 2)
        self.total_rotation += abs(dtheta)

        # 记录轨迹
        if self._recording:
            self._trajectory.append({
                'time': self._time,
                'x': self.x,
                'y': self.y,
                'theta': self.theta,
                'linear_velocity': linear_speed,
                'angular_velocity': angular_speed
            })

    def enable_trajectory_recording(self, enable=True):
        self._recording = enable
        if enable:
            self._trajectory = [{
                'time': 0.0,
                'x': self.x,
                'y': self.y,
                'theta': self.theta,
                'linear_velocity': 0.0,
                'angular_velocity': 0.0
            }]

    def get_trajectory(self):
        return self._trajectory.copy()


# ==================== 轨迹模拟函数 ====================
def simulate_square_trajectory():
    print("模拟正方形轨迹...")
    robot = DifferentialDriveKinematics()
    robot.enable_trajectory_recording(True)
    side_length = 2.0
    vertices = [(0, 0), (side_length, 0), (side_length, side_length), (0, side_length), (0, 0)]

    for i in range(len(vertices) - 1):
        start = vertices[i]
        end = vertices[i + 1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        target_angle = math.atan2(dy, dx)

        current_x, current_y, current_theta = robot.pose
        angle_error = ((target_angle - current_theta + math.pi) % (2 * math.pi)) - math.pi

        rotation_time = abs(angle_error) / (math.pi / 2)
        if rotation_time > 0:
            robot.move_with_velocity(0.0, angle_error / rotation_time, dt=rotation_time)

        distance = math.sqrt(dx ** 2 + dy ** 2)
        travel_time = distance / 0.5
        if travel_time > 0:
            robot.move_with_velocity(0.5, 0.0, dt=travel_time)

    print(f"正方形轨迹完成，边长: {side_length}m")
    print(f"总行驶距离: {robot.total_distance:.2f}m")
    return robot


def simulate_circular_trajectory(radius=1.0, revolutions=2):
    print(f"模拟圆形轨迹，半径: {radius}m，圈数: {revolutions}")
    robot = DifferentialDriveKinematics()
    robot.enable_trajectory_recording(True)

    circumference = 2 * math.pi * radius
    total_distance = circumference * revolutions
    linear_speed = 0.3
    angular_speed = linear_speed / radius
    total_time = total_distance / linear_speed

    steps = int(total_time / 0.1)
    for i in range(steps):
        robot.move_with_velocity(linear_speed, angular_speed, dt=0.1)

    print(f"圆形轨迹完成")
    print(f"理论周长: {circumference:.2f}m，实际行驶: {robot.total_distance:.2f}m")
    print(f"理论旋转: {2 * math.pi * revolutions:.2f}rad，实际旋转: {robot.total_rotation:.2f}rad")
    return robot


def simulate_figure_eight():
    print("模拟8字形轨迹...")
    robot = DifferentialDriveKinematics()
    robot.enable_trajectory_recording(True)

    radius = 1.0
    linear_speed = 0.3
    angular_speed = linear_speed / radius

    print("  第一个圆（顺时针）")
    steps_per_circle = int((2 * math.pi / angular_speed) / 0.1)
    for i in range(steps_per_circle):
        robot.move_with_velocity(linear_speed, angular_speed, dt=0.1)

    print("  第二个圆（逆时针）")
    for i in range(steps_per_circle):
        robot.move_with_velocity(linear_speed, -angular_speed, dt=0.1)

    print(f"8字形轨迹完成")
    print(f"总行驶距离: {robot.total_distance:.2f}m")
    return robot


def simulate_random_walk(steps=100):
    print(f"模拟随机行走，步数: {steps}")
    robot = DifferentialDriveKinematics()
    robot.enable_trajectory_recording(True)

    for i in range(steps):
        mode = random.choice(['straight', 'turn_left', 'turn_right', 'curve'])

        if mode == 'straight':
            speed = random.uniform(0.2, 0.5)
            robot.move_with_velocity(speed, 0.0, dt=0.2)
        elif mode == 'turn_left':
            robot.move_with_velocity(0.1, random.uniform(0.5, 1.0), dt=0.2)
        elif mode == 'turn_right':
            robot.move_with_velocity(0.1, random.uniform(-1.0, -0.5), dt=0.2)
        else:  # curve
            speed = random.uniform(0.2, 0.4)
            angular = random.uniform(-0.8, 0.8)
            robot.move_with_velocity(speed, angular, dt=0.2)

    print(f"随机行走完成")
    print(f"最终位姿: x={robot.x:.2f}m, y={robot.y:.2f}m, θ={math.degrees(robot.theta):.1f}°")
    print(f"总行驶距离: {robot.total_distance:.2f}m")
    return robot


def plot_comparison(robots, titles):
    """比较多个轨迹"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    colors = ['b', 'g', 'r', 'c', 'm', 'y']

    for idx, (robot, title) in enumerate(zip(robots, titles)):
        if idx >= 6:
            break

        trajectory = robot.get_trajectory()
        if not trajectory:
            continue

        xs = [p['x'] for p in trajectory]
        ys = [p['y'] for p in trajectory]
        linear_vels = [p['linear_velocity'] for p in trajectory]
        angular_vels = [p['angular_velocity'] for p in trajectory]

        # 1. 轨迹图
        ax1 = axes[0, 0]
        ax1.plot(xs, ys, color=colors[idx], linewidth=1.5, alpha=0.7, label=title)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('轨迹比较')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.axis('equal')

        # 2. X位置分布
        ax2 = axes[0, 1]
        ax2.hist(xs, bins=20, color=colors[idx], alpha=0.5, label=title)
        ax2.set_xlabel('X位置 (m)')
        ax2.set_ylabel('频率')
        ax2.set_title('X位置分布')
        ax2.legend()

        # 3. Y位置分布
        ax3 = axes[0, 2]
        ax3.hist(ys, bins=20, color=colors[idx], alpha=0.5, label=title)
        ax3.set_xlabel('Y位置 (m)')
        ax3.set_ylabel('频率')
        ax3.set_title('Y位置分布')
        ax3.legend()

        # 4. 速度分布
        ax4 = axes[1, 0]
        ax4.hist(linear_vels, bins=20, color=colors[idx], alpha=0.5, label=title)
        ax4.set_xlabel('线速度 (m/s)')
        ax4.set_ylabel('频率')
        ax4.set_title('线速度分布')
        ax4.legend()

        # 5. 角速度分布
        ax5 = axes[1, 1]
        ax5.hist(angular_vels, bins=20, color=colors[idx], alpha=0.5, label=title)
        ax5.set_xlabel('角速度 (rad/s)')
        ax5.set_ylabel('频率')
        ax5.set_title('角速度分布')
        ax5.legend()

        # 6. 统计信息
        ax6 = axes[1, 2]
        stats_text = (
            f'{title}\n'
            f'距离: {robot.total_distance:.2f}m\n'
            f'旋转: {math.degrees(robot.total_rotation):.1f}°\n'
            f'最大速度: {max(abs(v) for v in linear_vels):.2f}m/s\n'
            f'平均速度: {np.mean([abs(v) for v in linear_vels]):.2f}m/s'
        )
        ax6.text(0.5, 0.5, stats_text,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=ax6.transAxes,
                 fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax6.axis('off')

    plt.suptitle('轨迹模拟比较', fontsize=16)
    plt.tight_layout()
    plt.savefig('trajectory_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    print("=" * 60)
    print("轨迹模拟演示")
    print("=" * 60)

    robots = []
    titles = []

    print("\n1. 正方形轨迹")
    robot1 = simulate_square_trajectory()
    robots.append(robot1)
    titles.append("正方形")

    print("\n2. 圆形轨迹")
    robot2 = simulate_circular_trajectory(radius=1.0, revolutions=1)
    robots.append(robot2)
    titles.append("圆形")

    print("\n3. 8字形轨迹")
    robot3 = simulate_figure_eight()
    robots.append(robot3)
    titles.append("8字形")

    print("\n4. 随机行走")
    robot4 = simulate_random_walk(steps=50)
    robots.append(robot4)
    titles.append("随机行走")

    print("\n生成比较图表...")
    plot_comparison(robots, titles)

    print("\n" + "=" * 60)
    print("轨迹模拟完成!")
    print("比较图表已保存: trajectory_comparison.png")
    print("=" * 60)


if __name__ == "__main__":
    main()