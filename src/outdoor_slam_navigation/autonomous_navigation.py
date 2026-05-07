# autonomous_navigation.py
import numpy as np
from queue import PriorityQueue
from dataclasses import dataclass
from typing import List, Tuple, Optional
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Rectangle, Polygon
import matplotlib
import time
import threading
from scipy.spatial.transform import Rotation
import random
import os


# ==================== 设置中文字体 ====================
def setup_chinese_font():
    """设置中文字体支持"""
    # 尝试不同的方法设置中文字体
    try:
        # 方法1: 使用系统自带的中文字体
        font_names = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']

        # 获取系统中文字体
        system_fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
        chinese_fonts = [f for f in system_fonts if any(name in f for name in ['SimHei', 'YaHei', 'msyh', 'arial'])]

        if chinese_fonts:
            # 添加字体路径
            for font_path in chinese_fonts[:3]:  # 尝试前3个
                matplotlib.font_manager.fontManager.addfont(font_path)
                font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
                font_names.append(font_name)

        # 尝试设置字体
        for font_name in font_names:
            try:
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"已设置字体: {font_name}")
                return True
            except:
                continue

        # 如果都不行，使用默认字体显示英文
        print("警告: 未找到中文字体，将使用英文字体")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return False

    except Exception as e:
        print(f"字体设置错误: {e}")
        # 使用英文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return False


# 在程序开始前设置字体
setup_chinese_font()
@dataclass
class GridCell:
    x: int
    y: int
    cost: float = float('inf')
    parent: Optional[Tuple[int, int]] = None
    visited: bool = False


class AutonomousNavigation:
    """自主导航模块"""

    def __init__(self, config=None):
        self.config = config or self._default_config()

        # 地图相关
        self.occupancy_grid = None
        self.costmap = None
        self.resolution = 0.05  # 米/像素，提高分辨率

        # 路径规划
        self.global_path = None
        self.local_path = None

        # 控制参数
        self.current_speed = 0.0
        self.current_angular = 0.0
        self.target_speed = 0.0
        self.target_angular = 0.0
        self.max_acceleration = 0.5  # m/s²

        # 机器人状态
        self.robot_pose = np.eye(4)
        self.goal_pose = np.eye(4)
        self.goal_pose[0, 3] = 5.0  # 默认目标在X方向5米处
        self.goal_pose[1, 3] = 5.0  # Y方向5米处

        # 避障
        self.obstacle_buffer = []
        self.safety_distance = 0.5  # 米
        self.obstacles = []  # 障碍物列表

        # 导航状态
        self.navigation_active = False
        self.goal_reached = False
        self.path_found = False
        self.collision_warning = False

        # 可视化相关
        self.fig = None
        self.ax_map = None
        self.ax_path = None
        self.ax_info = None
        self.trajectory_history = []

        # 初始化障碍物环境
        self._initialize_environment()

        # 导航线程
        self.navigation_thread = None

    def _default_config(self):
        return {
            "planning": {
                "inflation_radius": 0.3,
                "path_smoothing": True,
                "replan_interval": 1.0,  # 重规划间隔缩短
                "goal_tolerance": 0.2,  # 目标容忍距离
            },
            "control": {
                "max_linear_speed": 0.5,  # 降低最大速度便于观察
                "max_angular_speed": 1.0,
                "lookahead_distance": 0.8,
                "pid_kp": 1.0,  # PID参数
                "pid_ki": 0.1,
                "pid_kd": 0.05,
            },
            "obstacle": {
                "max_obstacle_height": 0.5,
                "min_obstacle_height": 0.05,
                "inflation_layers": 3
            }
        }

    def _initialize_environment(self):
        """初始化模拟环境"""
        # 创建一些随机障碍物
        self.obstacles = [
            # 矩形障碍物 (x, y, width, height)
            (1.5, 1.5, 0.8, 0.3),
            (2.5, 3.0, 0.5, 1.0),
            (4.0, 2.0, 0.3, 0.8),
            (3.0, 4.5, 1.0, 0.4),
            (0.5, 3.5, 0.6, 0.6),
        ]

        # 设置随机目标点
        self.goal_pose = np.eye(4)
        self.goal_pose[0, 3] = random.uniform(3, 6)
        self.goal_pose[1, 3] = random.uniform(3, 6)

        print(f"目标点位置: ({self.goal_pose[0, 3]:.2f}, {self.goal_pose[1, 3]:.2f})")

    def start_navigation(self, start_pose=None, goal_pose=None):
        """开始自主导航"""
        if start_pose is not None:
            self.robot_pose = start_pose.copy()

        if goal_pose is not None:
            self.goal_pose = goal_pose.copy()

        self.navigation_active = True
        self.goal_reached = False
        self.path_found = False
        self.collision_warning = False

        print(f"开始导航: 起点({self.robot_pose[0, 3]:.2f}, {self.robot_pose[1, 3]:.2f}) -> "
              f"目标({self.goal_pose[0, 3]:.2f}, {self.goal_pose[1, 3]:.2f})")

        # 启动导航线程
        self.navigation_thread = threading.Thread(target=self._navigation_loop)
        self.navigation_thread.daemon = True
        self.navigation_thread.start()

        return True

    def stop_navigation(self):
        """停止导航"""
        self.navigation_active = False
        self.target_speed = 0.0
        self.target_angular = 0.0
        print("导航已停止")

    def _navigation_loop(self):
        """导航主循环"""
        last_replan_time = time.time()
        replan_interval = self.config["planning"]["replan_interval"]

        while self.navigation_active and not self.goal_reached:
            try:
                # 1. 更新感知
                self._update_perception()

                # 2. 检查是否到达目标
                if self._check_goal_reached():
                    self.goal_reached = True
                    print("成功到达目标点！")
                    self.target_speed = 0.0
                    self.target_angular = 0.0
                    break

                # 3. 碰撞检测
                if self._check_collision():
                    self.collision_warning = True
                    print("警告：检测到碰撞风险！")
                    # 执行紧急避障
                    self._emergency_avoidance()
                    continue
                else:
                    self.collision_warning = False

                # 4. 路径规划（按需重规划）
                current_time = time.time()
                if (not self.path_found or
                        (current_time - last_replan_time > replan_interval) or
                        self.global_path is None):

                    print("执行路径规划...")
                    if self.global_path_planning(self.robot_pose, self.goal_pose):
                        self.path_found = True
                        last_replan_time = current_time
                    else:
                        print("路径规划失败，尝试绕行...")
                        self._recover_from_planning_failure()
                        continue

                # 5. 局部路径规划
                if self.path_found and self.global_path is not None:
                    result = self.local_path_planning(
                        self.global_path, self.robot_pose, self.obstacles
                    )

                    if result is not None:
                        local_path, control_cmd = result
                        self.target_speed, self.target_angular = control_cmd

                        # 应用PID控制
                        self._apply_pid_control()
                    else:
                        print("局部路径规划失败")
                        self.target_speed = 0.0
                        self.target_angular = 0.0

                # 6. 更新机器人位置（模拟移动）
                self._update_robot_position()

                # 7. 更新轨迹历史
                robot_pos = self.robot_pose[:3, 3]
                self.trajectory_history.append((robot_pos[0], robot_pos[1]))
                if len(self.trajectory_history) > 200:
                    self.trajectory_history.pop(0)

                # 控制循环频率
                time.sleep(0.05)  # 20Hz

            except Exception as e:
                print(f"导航循环错误: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

        self.navigation_active = False
        print("导航循环结束")

    def _update_perception(self):
        """更新感知信息（模拟传感器数据）"""
        # 生成模拟点云数据
        point_cloud = self._generate_simulated_pointcloud()

        # 更新占据栅格
        self.update_occupancy_grid(point_cloud, self.robot_pose)

        return point_cloud

    def _generate_simulated_pointcloud(self):
        """生成模拟点云数据"""
        point_cloud = []

        # 添加障碍物点
        for obs in self.obstacles:
            x, y, w, h = obs
            # 在障碍物表面采样点
            num_points = 20
            for _ in range(num_points):
                px = x + random.uniform(-w / 2, w / 2)
                py = y + random.uniform(-h / 2, h / 2)
                # 转换为机器人坐标系
                robot_pos = self.robot_pose[:3, 3]
                dx = px - robot_pos[0]
                dy = py - robot_pos[1]
                distance = np.sqrt(dx ** 2 + dy ** 2)

                # 只添加在传感器范围内的点
                if distance < 3.0:  # 3米范围内
                    point_cloud.append([dx, dy, 0.1])

        # 添加一些随机噪声点
        for _ in range(10):
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(0.5, 3.0)
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            point_cloud.append([x, y, 0.1])

        return point_cloud

    def _check_goal_reached(self):
        """检查是否到达目标"""
        robot_pos = self.robot_pose[:3, 3]
        goal_pos = self.goal_pose[:3, 3]

        distance = np.linalg.norm(robot_pos[:2] - goal_pos[:2])
        tolerance = self.config["planning"]["goal_tolerance"]

        return distance < tolerance

    def _check_collision(self):
        """碰撞检测"""
        if self.occupancy_grid is None:
            return False

        robot_grid = self._pose_to_grid(self.robot_pose)
        grid_size = self.occupancy_grid.shape[0]

        # 检查机器人周围的栅格
        check_radius = int(self.safety_distance / self.resolution)

        for dx in range(-check_radius, check_radius + 1):
            for dy in range(-check_radius, check_radius + 1):
                nx, ny = robot_grid[0] + dx, robot_grid[1] + dy
                distance = np.sqrt(dx ** 2 + dy ** 2) * self.resolution

                if distance <= self.safety_distance:
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        if self.occupancy_grid[nx, ny] > 0.3:
                            return True

        return False

    def _emergency_avoidance(self):
        """紧急避障"""
        print("执行紧急避障...")

        # 后退并转向
        self.target_speed = -0.1  # 后退
        self.target_angular = random.choice([-0.5, 0.5])  # 随机转向

        # 应用控制
        self._apply_pid_control()
        self._update_robot_position()

        # 短暂避障后停止
        time.sleep(0.5)
        self.target_speed = 0.0
        self.target_angular = 0.0

        # 标记需要重新规划
        self.path_found = False

    def _recover_from_planning_failure(self):
        """从规划失败中恢复"""
        # 尝试随机移动以摆脱困境
        self.target_speed = 0.1
        self.target_angular = random.uniform(-1.0, 1.0)

        self._apply_pid_control()
        self._update_robot_position()

        time.sleep(0.3)
        self.target_speed = 0.0
        self.target_angular = 0.0

    def _apply_pid_control(self):
        """应用PID控制更新实际速度"""
        # 简化的速度控制
        acceleration = self.max_acceleration
        dt = 0.05

        # 线性速度控制
        speed_error = self.target_speed - self.current_speed
        self.current_speed += np.clip(speed_error, -acceleration * dt, acceleration * dt)

        # 角速度控制
        angular_error = self.target_angular - self.current_angular
        self.current_angular += np.clip(angular_error, -2.0 * dt, 2.0 * dt)

    def _update_robot_position(self):
        """更新机器人位置（模拟运动）"""
        # 获取当前位姿
        pos = self.robot_pose[:3, 3]
        rot = self.robot_pose[:3, :3]

        # 获取偏航角
        r = Rotation.from_matrix(rot)
        yaw = r.as_euler('zyx')[0]

        # 更新位置
        dt = 0.05
        dx = self.current_speed * np.cos(yaw) * dt
        dy = self.current_speed * np.sin(yaw) * dt
        dtheta = self.current_angular * dt

        # 更新位置
        self.robot_pose[0, 3] += dx
        self.robot_pose[1, 3] += dy

        # 更新朝向
        new_yaw = yaw + dtheta
        new_rot = Rotation.from_euler('z', new_yaw).as_matrix()
        self.robot_pose[:3, :3] = new_rot

        # 确保位置在合理范围内
        self.robot_pose[0, 3] = np.clip(self.robot_pose[0, 3], -2.0, 8.0)
        self.robot_pose[1, 3] = np.clip(self.robot_pose[1, 3], -2.0, 8.0)

    # ============== 原有的路径规划方法 ==============

    def update_occupancy_grid(self, point_cloud, robot_pose):
        """从点云更新占据栅格地图"""
        if point_cloud is None or len(point_cloud) == 0:
            return

        # 初始化栅格地图
        grid_size = 200  # 200x200网格，提高分辨率
        if self.occupancy_grid is None:
            self.occupancy_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
            self.costmap = np.ones((grid_size, grid_size), dtype=np.float32)

        # 清空地图
        self.occupancy_grid.fill(0.0)
        self.costmap.fill(1.0)

        # 机器人位置（地图中心）
        robot_x, robot_y = grid_size // 2, grid_size // 2

        # 将点云转换到地图坐标系
        for point in point_cloud:
            x, y, z = point[:3]

            # 过滤障碍物高度
            if self.config["obstacle"]["min_obstacle_height"] < z < self.config["obstacle"]["max_obstacle_height"]:
                # 转换为栅格坐标
                grid_x = int(robot_x + x / self.resolution)
                grid_y = int(robot_y + y / self.resolution)

                # 检查边界
                if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                    self.occupancy_grid[grid_x, grid_y] = 1.0

        # 添加静态障碍物
        self._add_static_obstacles_to_grid()

        # 生成代价地图
        self._generate_costmap()

    def _add_static_obstacles_to_grid(self):
        """将静态障碍物添加到栅格地图"""
        if self.occupancy_grid is None:
            return

        grid_size = self.occupancy_grid.shape[0]
        robot_x, robot_y = grid_size // 2, grid_size // 2
        robot_pos = self.robot_pose[:3, 3]

        for obs in self.obstacles:
            x, y, w, h = obs

            # 转换为相对于机器人的坐标
            dx = x - robot_pos[0]
            dy = y - robot_pos[1]

            # 转换为栅格坐标
            grid_x = int(robot_x + dx / self.resolution)
            grid_y = int(robot_y + dy / self.resolution)

            # 计算障碍物在栅格中的大小
            grid_w = int(w / self.resolution)
            grid_h = int(h / self.resolution)

            # 填充障碍物区域
            for i in range(-grid_w // 2, grid_w // 2 + 1):
                for j in range(-grid_h // 2, grid_h // 2 + 1):
                    nx, ny = grid_x + i, grid_y + j
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        self.occupancy_grid[nx, ny] = 1.0

    def _generate_costmap(self):
        """生成代价地图（包含障碍物膨胀）"""
        if self.occupancy_grid is None:
            return

        grid_size = self.occupancy_grid.shape[0]
        inflation_radius = int(self.config["planning"]["inflation_radius"] / self.resolution)

        # 创建膨胀后的障碍物地图
        inflated_grid = self.occupancy_grid.copy()

        for i in range(grid_size):
            for j in range(grid_size):
                if self.occupancy_grid[i, j] > 0.5:  # 障碍物
                    # 膨胀操作
                    for di in range(-inflation_radius, inflation_radius + 1):
                        for dj in range(-inflation_radius, inflation_radius + 1):
                            distance = np.sqrt(di ** 2 + dj ** 2)
                            if distance <= inflation_radius:
                                ni, nj = i + di, j + dj
                                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                                    # 距离越近，代价越高
                                    cost = 1.0 - (distance / inflation_radius) * 0.8
                                    inflated_grid[ni, nj] = max(inflated_grid[ni, nj], cost)

        self.costmap = 1.0 - inflated_grid  # 转换为通行代价（0为高代价，1为低代价）

    def global_path_planning(self, start_pose, goal_pose):
        """全局路径规划（A*算法）"""
        # 转换为栅格坐标
        start_grid = self._pose_to_grid(start_pose)
        goal_grid = self._pose_to_grid(goal_pose)

        # 检查目标是否可达
        if not self._is_cell_free(goal_grid[0], goal_grid[1]):
            print("目标点被障碍物占据")
            return None

        # A*算法
        open_set = PriorityQueue()
        closed_set = set()

        # 初始化起始节点
        start_node = GridCell(start_grid[0], start_grid[1], 0, None)
        start_f = self._heuristic(start_grid, goal_grid)
        open_set.put((start_f, start_node))

        while not open_set.empty():
            current_f, current_node = open_set.get()

            # 到达目标
            if (current_node.x, current_node.y) == goal_grid:
                self.global_path = self._reconstruct_path(current_node)
                return self.global_path

            closed_set.add((current_node.x, current_node.y))

            # 扩展邻居节点
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (-1, 1), (1, -1), (-1, -1)]:
                nx, ny = current_node.x + dx, current_node.y + dy

                # 检查边界和障碍物
                if not self._is_cell_free(nx, ny):
                    continue

                # 计算移动代价
                move_cost = 1.0 if dx == 0 or dy == 0 else 1.414  # 对角线代价
                cell_cost = self.costmap[nx, ny]
                total_cost = current_node.cost + move_cost * (2.0 - cell_cost)  # 代价高的区域惩罚

                # 检查是否已在closed_set
                if (nx, ny) in closed_set:
                    continue

                # 计算启发值
                g = total_cost
                h = self._heuristic((nx, ny), goal_grid)
                f = g + h

                # 添加到open_set
                neighbor_node = GridCell(nx, ny, total_cost, (current_node.x, current_node.y))
                open_set.put((f, neighbor_node))

        print("找不到路径")
        return None

    def _pose_to_grid(self, pose):
        """将位姿转换为栅格坐标"""
        if self.occupancy_grid is None:
            return 0, 0

        grid_size = self.occupancy_grid.shape[0]
        robot_x, robot_y = grid_size // 2, grid_size // 2

        # 从位姿中提取位置
        position = pose[:3, 3]

        grid_x = int(robot_x + position[0] / self.resolution)
        grid_y = int(robot_y + position[1] / self.resolution)

        return grid_x, grid_y

    def _is_cell_free(self, x, y):
        """检查栅格是否可通行"""
        if self.occupancy_grid is None:
            return False

        grid_size = self.occupancy_grid.shape[0]

        # 检查边界
        if x < 0 or x >= grid_size or y < 0 or y >= grid_size:
            return False

        # 检查障碍物
        if self.occupancy_grid[x, y] > 0.5:
            return False

        # 检查代价（太低表示太靠近障碍物）
        if self.costmap[x, y] < 0.3:
            return False

        return True

    def _heuristic(self, a, b):
        """启发函数（欧几里得距离）"""
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def _reconstruct_path(self, goal_node):
        """重构路径"""
        path = []
        current = goal_node

        while current is not None:
            # 转换为世界坐标
            world_x, world_y = self._grid_to_world(current.x, current.y)
            path.append((world_x, world_y))
            current = current.parent

        path.reverse()

        # 路径平滑
        if self.config["planning"]["path_smoothing"]:
            path = self._smooth_path(path)

        return np.array(path)

    def _grid_to_world(self, grid_x, grid_y):
        """将栅格坐标转换为世界坐标"""
        if self.occupancy_grid is None:
            return 0.0, 0.0

        grid_size = self.occupancy_grid.shape[0]
        robot_x, robot_y = grid_size // 2, grid_size // 2

        world_x = (grid_x - robot_x) * self.resolution
        world_y = (grid_y - robot_y) * self.resolution

        return world_x, world_y

    def _smooth_path(self, path, weight_data=0.1, weight_smooth=0.3, tolerance=0.00001):
        """路径平滑（梯度下降）"""
        if len(path) < 3:
            return path

        smoothed = np.array(path).copy()

        change = tolerance
        while change >= tolerance:
            change = 0.0

            for i in range(1, len(path) - 1):
                original = np.array(path[i])
                current = smoothed[i]

                # 数据项（保持接近原始路径）
                data_term = original - current

                # 平滑项（保持相邻点间距均匀）
                smooth_term = (smoothed[i - 1] + smoothed[i + 1] - 2 * current)

                # 更新
                update = weight_data * data_term + weight_smooth * smooth_term
                smoothed[i] += update

                change += np.linalg.norm(update)

        return smoothed.tolist()

    def local_path_planning(self, global_path, robot_pose, obstacles=None):
        """局部路径规划（动态窗口方法）"""
        if global_path is None or len(global_path) == 0:
            return None

        # 找到路径上最近的点
        robot_pos = robot_pose[:3, 3]
        distances = np.linalg.norm(global_path - robot_pos[:2], axis=1)
        nearest_idx = np.argmin(distances)

        # 向前看一段距离
        lookahead_distance = self.config["control"]["lookahead_distance"]
        target_idx = nearest_idx

        for i in range(nearest_idx, len(global_path)):
            if np.linalg.norm(global_path[i] - robot_pos[:2]) >= lookahead_distance:
                target_idx = i
                break

        if target_idx >= len(global_path):
            target_idx = len(global_path) - 1

        target_point = global_path[target_idx]

        # 动态窗口方法
        best_v, best_w = self._dynamic_window_approach(robot_pose, target_point, obstacles)

        # 生成局部路径
        local_path = []
        dt = 0.1

        for i in range(10):  # 预测10步
            t = i * dt
            x = robot_pos[0] + best_v * np.cos(robot_pos[2]) * t
            y = robot_pos[1] + best_v * np.sin(robot_pos[2]) * t
            theta = robot_pos[2] + best_w * t

            local_path.append((x, y, theta))

        self.local_path = np.array(local_path)
        return self.local_path, (best_v, best_w)

    def _dynamic_window_approach(self, robot_pose, target, obstacles):
        """动态窗口方法"""
        # 机器人状态
        x, y, theta = self._extract_robot_state(robot_pose)

        # 速度范围
        v_min, v_max = 0.0, self.config["control"]["max_linear_speed"]
        w_min, w_max = -self.config["control"]["max_angular_speed"], self.config["control"]["max_angular_speed"]

        # 生成速度样本
        v_samples = np.linspace(v_min, v_max, 10)
        w_samples = np.linspace(w_min, w_max, 20)

        best_score = -float('inf')
        best_v, best_w = 0.0, 0.0

        for v in v_samples:
            for w in w_samples:
                # 模拟轨迹
                trajectory = self._simulate_trajectory(x, y, theta, v, w, 1.0, 0.1)

                # 计算分数
                goal_score = self._goal_score(trajectory, target)
                obstacle_score = self._obstacle_score(trajectory, obstacles)
                speed_score = v / v_max  # 鼓励移动

                total_score = 0.5 * goal_score + 0.3 * obstacle_score + 0.2 * speed_score

                if total_score > best_score:
                    best_score = total_score
                    best_v, best_w = v, w

        return best_v, best_w

    def _extract_robot_state(self, pose):
        """从位姿中提取机器人状态"""
        position = pose[:3, 3]
        # 从旋转矩阵中提取偏航角
        r = Rotation.from_matrix(pose[:3, :3])
        yaw = r.as_euler('zyx')[0]

        return position[0], position[1], yaw

    def _simulate_trajectory(self, x, y, theta, v, w, time_horizon, dt):
        """模拟轨迹"""
        trajectory = []
        t = 0.0

        while t < time_horizon:
            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt
            theta += w * dt

            trajectory.append((x, y, theta))
            t += dt

        return np.array(trajectory)

    def _goal_score(self, trajectory, target):
        """目标得分"""
        if len(trajectory) == 0:
            return 0.0

        # 使用轨迹终点的距离
        final_pos = trajectory[-1, :2]
        distance = np.linalg.norm(final_pos - target)

        # 距离越小，分数越高
        return 1.0 / (1.0 + distance)

    def _obstacle_score(self, trajectory, obstacles):
        """障碍物得分"""
        if obstacles is None or len(obstacles) == 0:
            return 1.0

        min_distance = float('inf')

        for point in trajectory[:, :2]:
            for obstacle in obstacles:
                ox, oy, ow, oh = obstacle
                # 计算到障碍物边界的最小距离
                dx = max(ox - ow / 2 - point[0], 0, point[0] - ox - ow / 2)
                dy = max(oy - oh / 2 - point[1], 0, point[1] - oy - oh / 2)
                distance = np.sqrt(dx * dx + dy * dy)
                min_distance = min(min_distance, distance)

        # 距离越大，分数越高
        if min_distance < self.safety_distance:
            return 0.0
        else:
            return min(1.0, min_distance / 5.0)

    def check_collision(self, point_cloud=None, safety_margin=0.3):
        """碰撞检测"""
        # 使用内部的碰撞检测方法
        return self._check_collision()

    def execute_emergency_stop(self):
        """执行紧急停止"""
        print("执行紧急停止！")
        self.target_speed = 0.0
        self.target_angular = 0.0
        return True

    # ============== 可视化方法 ==============

    def setup_visualization(self):
        """设置可视化窗口"""
        plt.ion()  # 打开交互模式

        self.fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                               width_ratios=[1.5, 1],
                               height_ratios=[1, 1])

        # 创建子图
        self.ax_map = self.fig.add_subplot(gs[:, 0])  # 左侧：占据栅格地图
        self.ax_path = self.fig.add_subplot(gs[0, 1])  # 右上：路径规划
        self.ax_info = self.fig.add_subplot(gs[1, 1])  # 右下：信息显示

        # 设置标题
        self.fig.suptitle('自主导航系统 - 实时演示', fontsize=16, fontweight='bold')

        plt.tight_layout()
        print("可视化窗口已初始化")

    def update_visualization(self):
        """更新可视化显示"""
        if not hasattr(self, 'fig'):
            self.setup_visualization()

        # 清空画布
        self.ax_map.clear()
        self.ax_path.clear()
        self.ax_info.clear()
        self.ax_info.axis('off')

        # 获取位置信息
        robot_pos = self.robot_pose[:3, 3]
        goal_pos = self.goal_pose[:3, 3]

        # 1. 绘制占据栅格地图和场景
        self._draw_scene_map()

        # 2. 绘制路径规划
        self._draw_path_planning()

        # 3. 更新信息显示
        self._update_info_display()

        # 刷新显示
        plt.draw()
        plt.pause(0.001)

    def _draw_scene_map(self):
        """绘制场景地图"""
        robot_pos = self.robot_pose[:3, 3]
        goal_pos = self.goal_pose[:3, 3]

        # 绘制障碍物
        for obs in self.obstacles:
            x, y, w, h = obs
            rect = Rectangle((x - w / 2, y - h / 2), w, h,
                             facecolor='red', alpha=0.6,
                             edgecolor='darkred', linewidth=2)
            self.ax_map.add_patch(rect)

        # 绘制机器人
        robot_circle = Circle((robot_pos[0], robot_pos[1]),
                              self.safety_distance,
                              color='blue', alpha=0.3, label='机器人')
        self.ax_map.add_patch(robot_circle)

        # 绘制机器人方向
        r = Rotation.from_matrix(self.robot_pose[:3, :3])
        yaw = r.as_euler('zyx')[0]
        arrow_length = self.safety_distance * 1.5
        dx = arrow_length * np.cos(yaw)
        dy = arrow_length * np.sin(yaw)
        self.ax_map.arrow(robot_pos[0], robot_pos[1], dx, dy,
                          head_width=0.1, head_length=0.15,
                          fc='darkblue', ec='darkblue')

        # 绘制目标点
        goal_circle = Circle((goal_pos[0], goal_pos[1]),
                             self.config["planning"]["goal_tolerance"],
                             color='green', alpha=0.4, label='目标区域')
        self.ax_map.add_patch(goal_circle)
        self.ax_map.plot(goal_pos[0], goal_pos[1], 'g*',
                         markersize=20, label='目标点')

        # 绘制轨迹
        if len(self.trajectory_history) > 1:
            trajectory = np.array(self.trajectory_history)
            self.ax_map.plot(trajectory[:, 0], trajectory[:, 1],
                             'blue', linewidth=2, alpha=0.6, label='轨迹')

        # 设置地图范围
        self.ax_map.set_xlim(-1, 7)
        self.ax_map.set_ylim(-1, 7)
        self.ax_map.set_xlabel('X (米)')
        self.ax_map.set_ylabel('Y (米)')
        self.ax_map.grid(True, alpha=0.3)
        self.ax_map.set_title('导航场景', fontsize=12)
        self.ax_map.legend(loc='upper right')
        self.ax_map.set_aspect('equal')

    def _draw_path_planning(self):
        """绘制路径规划"""
        robot_pos = self.robot_pose[:3, 3]
        goal_pos = self.goal_pose[:3, 3]

        # 绘制障碍物
        for obs in self.obstacles:
            x, y, w, h = obs
            rect = Rectangle((x - w / 2, y - h / 2), w, h,
                             facecolor='red', alpha=0.3,
                             edgecolor='darkred', linewidth=1)
            self.ax_path.add_patch(rect)

        # 绘制机器人
        self.ax_path.plot(robot_pos[0], robot_pos[1], 'bo',
                          markersize=12, label='机器人')

        # 绘制目标
        self.ax_path.plot(goal_pos[0], goal_pos[1], 'g*',
                          markersize=20, label='目标')

        # 绘制全局路径
        if self.global_path is not None and len(self.global_path) > 0:
            self.ax_path.plot(self.global_path[:, 0], self.global_path[:, 1],
                              'green', linewidth=3, alpha=0.7, label='全局路径')

            # 绘制路径点
            self.ax_path.scatter(self.global_path[:, 0], self.global_path[:, 1],
                                 color='green', s=30, alpha=0.5)

        # 绘制局部路径
        if self.local_path is not None and len(self.local_path) > 0:
            self.ax_path.plot(self.local_path[:, 0], self.local_path[:, 1],
                              'blue', linewidth=2, linestyle='--', label='局部路径')

            # 绘制预测轨迹
            for i in range(0, len(self.local_path), 2):
                x, y, theta = self.local_path[i]
                self.ax_path.plot(x, y, 'b.', markersize=8)

        # 绘制轨迹
        if len(self.trajectory_history) > 1:
            trajectory = np.array(self.trajectory_history)
            self.ax_path.plot(trajectory[:, 0], trajectory[:, 1],
                              'blue', linewidth=1.5, alpha=0.5, label='历史轨迹')

        # 设置图形属性
        self.ax_path.set_xlabel('X (米)')
        self.ax_path.set_ylabel('Y (米)')
        self.ax_path.grid(True, alpha=0.3)
        self.ax_path.set_title('路径规划', fontsize=12)
        self.ax_path.legend(loc='upper right', fontsize=8)

        # 自动调整视图范围
        all_x = [robot_pos[0], goal_pos[0]]
        all_y = [robot_pos[1], goal_pos[1]]

        if self.global_path is not None:
            all_x.extend(self.global_path[:, 0])
            all_y.extend(self.global_path[:, 1])

        if all_x:
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            margin = 1.0
            self.ax_path.set_xlim(x_min - margin, x_max + margin)
            self.ax_path.set_ylim(y_min - margin, y_max + margin)

    def _update_info_display(self):
        """更新信息显示"""
        robot_pos = self.robot_pose[:3, 3]
        goal_pos = self.goal_pose[:3, 3]

        r = Rotation.from_matrix(self.robot_pose[:3, :3])
        yaw = r.as_euler('zyx')[0]

        # 计算到目标的距离
        distance_to_goal = np.linalg.norm(robot_pos[:2] - goal_pos[:2])

        # 创建信息文本
        info_text = [
            "=== 导航状态 ===",
            f"运行状态: {'活跃' if self.navigation_active else '停止'}",
            f"目标到达: {'是' if self.goal_reached else '否'}",
            f"路径找到: {'是' if self.path_found else '否'}",
            f"碰撞警告: {'是' if self.collision_warning else '否'}",
            "",
            "=== 机器人状态 ===",
            f"位置: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f})",
            f"朝向: {np.degrees(yaw):.1f}°",
            f"速度: {self.current_speed:.2f} m/s",
            f"角速度: {np.degrees(self.current_angular):.1f} °/s",
            "",
            "=== 目标信息 ===",
            f"目标位置: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})",
            f"目标距离: {distance_to_goal:.2f} 米",
            f"目标容差: {self.config['planning']['goal_tolerance']} 米",
            "",
            "=== 控制指令 ===",
            f"目标速度: {self.target_speed:.2f} m/s",
            f"目标角速: {self.target_angular:.2f} rad/s",
        ]

        # 显示文本
        for i, text in enumerate(info_text):
            self.ax_info.text(0.05, 0.98 - i * 0.045, text,
                              transform=self.ax_info.transAxes,
                              fontsize=9, verticalalignment='top',
                              fontfamily='monospace')

        # 状态指示灯
        status_color = 'green' if self.navigation_active else 'red'
        status_text = '运行中' if self.navigation_active else '已停止'

        status_circle = Circle((0.15, 0.05), 0.03,
                               transform=self.ax_info.transAxes,
                               facecolor=status_color, alpha=0.8)
        self.ax_info.add_patch(status_circle)
        self.ax_info.text(0.2, 0.05, status_text,
                          transform=self.ax_info.transAxes,
                          fontsize=10, verticalalignment='center')

        # 进度条
        progress = 1.0 - min(1.0, distance_to_goal / 10.0)
        self.ax_info.add_patch(Rectangle((0.1, 0.12), 0.8, 0.04,
                                         transform=self.ax_info.transAxes,
                                         facecolor='lightgray',
                                         edgecolor='black'))

        progress_color = 'green' if progress > 0.7 else 'yellow' if progress > 0.3 else 'red'
        self.ax_info.add_patch(Rectangle((0.1, 0.12), 0.8 * progress, 0.04,
                                         transform=self.ax_info.transAxes,
                                         facecolor=progress_color,
                                         alpha=0.7))

        self.ax_info.text(0.5, 0.14, f"进度: {progress * 100:.1f}%",
                          transform=self.ax_info.transAxes,
                          fontsize=10, fontweight='bold',
                          horizontalalignment='center')

        self.ax_info.set_title('系统信息', fontsize=12)

    def run_autonomous_navigation_demo(self):
        """运行自主导航演示"""
        print("=== 自主导航系统演示 ===")
        print("正在启动系统...")

        # 设置可视化
        self.setup_visualization()

        # 设置随机起点
        start_pose = np.eye(4)
        start_pose[0, 3] = 0.0
        start_pose[1, 3] = 0.0

        # 开始导航
        self.start_navigation(start_pose, self.goal_pose)

        try:
            # 主循环：更新可视化
            while self.navigation_active:
                self.update_visualization()
                time.sleep(0.05)  # 20Hz更新频率

                # 如果到达目标，短暂暂停后设置新目标
                if self.goal_reached and self.navigation_active:
                    print("已到达目标，正在设置新目标...")
                    time.sleep(1.0)

                    # 设置新随机目标
                    new_goal = np.eye(4)
                    new_goal[0, 3] = random.uniform(2, 6)
                    new_goal[1, 3] = random.uniform(2, 6)

                    print(f"新目标: ({new_goal[0, 3]:.2f}, {new_goal[1, 3]:.2f})")
                    self.goal_pose = new_goal
                    self.goal_reached = False
                    self.path_found = False

            print("演示完成")

        except KeyboardInterrupt:
            print("\n用户中断")
            self.stop_navigation()

        finally:
            plt.ioff()
            plt.show()
            print("程序结束")



if __name__ == "__main__":
    # 创建导航系统
    nav_system = AutonomousNavigation()

    # 运行自主导航演示
    nav_system.run_autonomous_navigation_demo()