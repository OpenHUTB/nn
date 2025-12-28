# simple_autonomous_demo.py
import numpy as np
import matplotlib.pyplot as plt
import time
from autonomous_navigation import AutonomousNavigation
import random


class SimpleSimulator:
    """简化的自主导航模拟器"""

    def __init__(self):
        # 创建模拟环境
        self.env_size = 50  # 米
        self.resolution = 0.1  # 米/像素
        self.grid_size = int(self.env_size / self.resolution)

        # 初始化地图
        self.create_environment()

        # 初始化导航模块
        self.navigator = AutonomousNavigation()

        # 模拟机器人状态
        self.robot_pose = np.eye(4)
        self.robot_pose[0, 3] = 5.0  # 起始位置 x
        self.robot_pose[1, 3] = 5.0  # 起始位置 y

        # 目标位置
        self.goal_pose = np.eye(4)
        self.goal_pose[0, 3] = 45.0
        self.goal_pose[1, 3] = 45.0

        # 轨迹记录
        self.trajectory = []
        self.obstacles_hit = 0

        # 可视化设置
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))

    def create_environment(self):
        """创建模拟环境"""
        # 初始化占据栅格
        self.occupancy_grid = np.zeros((self.grid_size, self.grid_size))
        self.costmap = np.ones((self.grid_size, self.grid_size))

        # 添加障碍物
        self.add_obstacles()

        # 生成代价地图
        self.generate_costmap()

    def add_obstacles(self):
        """添加模拟障碍物"""
        # 随机障碍物
        num_obstacles = 15
        for _ in range(num_obstacles):
            ox = random.randint(10, self.grid_size - 10)
            oy = random.randint(10, self.grid_size - 10)
            radius = random.randint(3, 8)

            for i in range(ox - radius, ox + radius):
                for j in range(oy - radius, oy + radius):
                    if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
                        distance = np.sqrt((i - ox) ** 2 + (j - oy) ** 2)
                        if distance <= radius:
                            self.occupancy_grid[i, j] = 1.0

        # 添加墙壁边界
        wall_thickness = 2
        self.occupancy_grid[:wall_thickness, :] = 1.0
        self.occupancy_grid[-wall_thickness:, :] = 1.0
        self.occupancy_grid[:, :wall_thickness] = 1.0
        self.occupancy_grid[:, -wall_thickness:] = 1.0

        # 添加走廊
        for i in range(15, 35):
            for j in range(20, 25):
                if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
                    self.occupancy_grid[i, j] = 1.0

        for i in range(20, 25):
            for j in range(15, 35):
                if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
                    self.occupancy_grid[i, j] = 1.0

    def generate_costmap(self):
        """生成代价地图"""
        inflation_radius = int(0.3 / self.resolution)

        inflated_grid = self.occupancy_grid.copy()
        grid_size = self.grid_size

        for i in range(grid_size):
            for j in range(grid_size):
                if self.occupancy_grid[i, j] > 0.5:
                    for di in range(-inflation_radius, inflation_radius + 1):
                        for dj in range(-inflation_radius, inflation_radius + 1):
                            distance = np.sqrt(di ** 2 + dj ** 2)
                            if distance <= inflation_radius:
                                ni, nj = i + di, j + dj
                                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                                    cost = 1.0 - (distance / inflation_radius) * 0.8
                                    inflated_grid[ni, nj] = max(inflated_grid[ni, nj], cost)

        self.costmap = 1.0 - inflated_grid

    def simulate_lidar(self):
        """模拟LiDAR扫描"""
        num_points = 360
        max_range = 10.0  # 米

        points = []
        robot_x = self.robot_pose[0, 3]
        robot_y = self.robot_pose[1, 3]

        for angle in np.linspace(0, 2 * np.pi, num_points):
            # 射线追踪
            for r in np.linspace(0.1, max_range, 50):
                x = robot_x + r * np.cos(angle)
                y = robot_y + r * np.sin(angle)

                # 转换为栅格坐标
                grid_x = int(x / self.resolution)
                grid_y = int(y / self.resolution)

                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    if self.occupancy_grid[grid_x, grid_y] > 0.5:
                        # 添加一些噪声
                        x += random.uniform(-0.05, 0.05)
                        y += random.uniform(-0.05, 0.05)
                        z = random.uniform(-0.2, 0.2)
                        points.append([x - robot_x, y - robot_y, z])
                        break

        return np.array(points)

    def update_navigation(self):
        """更新导航系统"""
        # 模拟LiDAR数据
        point_cloud = self.simulate_lidar()

        # 更新占据栅格地图
        self.navigator.occupancy_grid = self.occupancy_grid
        self.navigator.costmap = self.costmap

        # 碰撞检测
        if self.navigator.check_collision(point_cloud, safety_margin=0.5):
            self.obstacles_hit += 1
            print(f"碰撞检测！总碰撞次数: {self.obstacles_hit}")

            # 后退并转向
            self.robot_pose[0, 3] -= 0.3 * np.cos(self.robot_pose[2, 0])
            self.robot_pose[1, 3] -= 0.3 * np.sin(self.robot_pose[2, 0])

            # 随机转向
            turn_angle = random.uniform(-np.pi / 4, np.pi / 4)
            self.robot_pose[:2, :2] = self.rotate_matrix(turn_angle)
            return False

        # 路径规划
        if not hasattr(self, 'global_path') or self.global_path is None:
            print("进行全局路径规划...")
            self.global_path = self.navigator.global_path_planning(
                self.robot_pose, self.goal_pose
            )

        if self.global_path is not None and len(self.global_path) > 0:
            # 局部路径规划
            local_path, control_cmd = self.navigator.local_path_planning(
                self.global_path, self.robot_pose, point_cloud
            )

            if control_cmd is not None:
                v, w = control_cmd

                # 更新机器人位姿
                dt = 0.1
                self.robot_pose[0, 3] += v * np.cos(self.robot_pose[2, 0]) * dt
                self.robot_pose[1, 3] += v * np.sin(self.robot_pose[2, 0]) * dt
                self.robot_pose[:2, :2] = self.rotate_matrix(w * dt)

        return True

    def rotate_matrix(self, angle):
        """创建2D旋转矩阵"""
        return np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])

    def visualize(self):
        """可视化当前状态"""
        self.ax1.clear()
        self.ax2.clear()

        # 1. 显示地图和路径
        self.ax1.imshow(self.costmap.T, cmap='gray', origin='lower',
                        extent=[0, self.env_size, 0, self.env_size])

        # 显示机器人位置
        robot_x, robot_y = self.robot_pose[0, 3], self.robot_pose[1, 3]
        self.ax1.plot(robot_x, robot_y, 'ro', markersize=10, label='Robot')

        # 显示目标位置
        goal_x, goal_y = self.goal_pose[0, 3], self.goal_pose[1, 3]
        self.ax1.plot(goal_x, goal_y, 'g*', markersize=15, label='Goal')

        # 显示全局路径
        if hasattr(self, 'global_path') and self.global_path is not None:
            self.ax1.plot(self.global_path[:, 0], self.global_path[:, 1],
                          'b-', linewidth=2, alpha=0.7, label='Global Path')

        # 显示轨迹
        if len(self.trajectory) > 0:
            traj = np.array(self.trajectory)
            self.ax1.plot(traj[:, 0], traj[:, 1], 'y-', linewidth=1, alpha=0.5, label='Trajectory')

        self.ax1.set_xlabel('X (m)')
        self.ax1.set_ylabel('Y (m)')
        self.ax1.set_title('Navigation Map')
        self.ax1.legend()
        self.ax1.grid(True)
        self.ax1.axis('equal')

        # 2. 显示代价地图热图
        costmap_display = self.ax2.imshow(self.costmap.T, cmap='viridis',
                                          origin='lower', vmin=0, vmax=1,
                                          extent=[0, self.env_size, 0, self.env_size])
        self.ax2.plot(robot_x, robot_y, 'ro', markersize=10)
        self.ax2.set_xlabel('X (m)')
        self.ax2.set_ylabel('Y (m)')
        self.ax2.set_title('Cost Map')
        self.ax2.grid(True)

        plt.colorbar(costmap_display, ax=self.ax2, label='Cost (0=high, 1=low)')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

    def run(self, max_steps=500):
        """运行模拟"""
        print("开始自主导航模拟...")
        print(f"起始位置: ({self.robot_pose[0, 3]:.1f}, {self.robot_pose[1, 3]:.1f})")
        print(f"目标位置: ({self.goal_pose[0, 3]:.1f}, {self.goal_pose[1, 3]:.1f})")

        for step in range(max_steps):
            # 记录当前位置
            self.trajectory.append([self.robot_pose[0, 3], self.robot_pose[1, 3]])

            # 更新导航
            success = self.update_navigation()

            # 可视化
            if step % 5 == 0:
                self.visualize()

            # 检查是否到达目标
            distance_to_goal = np.sqrt(
                (self.robot_pose[0, 3] - self.goal_pose[0, 3]) ** 2 +
                (self.robot_pose[1, 3] - self.goal_pose[1, 3]) ** 2
            )

            if distance_to_goal < 1.0:
                print(f"\n✓ 成功到达目标！")
                print(f"总步数: {step}")
                print(f"总碰撞次数: {self.obstacles_hit}")
                print(f"最终位置误差: {distance_to_goal:.2f} 米")
                break

            # 显示进度
            if step % 50 == 0:
                print(f"步数: {step}, 距离目标: {distance_to_goal:.1f}米")

            time.sleep(0.05)

        # 显示最终结果
        self.visualize()
        plt.ioff()

        # 保存结果
        self.fig.savefig('navigation_result.png', dpi=150)
        print(f"\n导航结果已保存到 'navigation_result.png'")

        # 显示总结
        print("\n=== 模拟总结 ===")
        print(f"总行驶距离: {len(self.trajectory) * 0.1:.1f} 米")
        print(f"碰撞次数: {self.obstacles_hit}")
        print(f"路径规划次数: 1")

        plt.show()


def main():
    """主函数"""
    simulator = SimpleSimulator()
    simulator.run(max_steps=500)


if __name__ == "__main__":
    main()