"""
机械臂轨迹规划与优化可视化系统
完整版本：包含RRT、PRM、A*等算法，以及丰富的可视化功能
Author: AI Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from scipy.spatial import KDTree, cKDTree
from scipy.interpolate import interp1d, splprep, splev
from scipy.optimize import minimize, Bounds
import networkx as nx
import heapq
import time
import json
import pickle
from queue import PriorityQueue
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. 路径规划算法实现
# ============================================================================

class RRTPlanner:
    """快速扩展随机树(RRT)路径规划算法"""

    def __init__(self, start, goal, obstacles, bounds, step_size=0.1, max_iter=5000):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles
        self.bounds = bounds  # [xmin, xmax, ymin, ymax, zmin, zmax]
        self.step_size = step_size
        self.max_iter = max_iter

        self.nodes = [self.start]
        self.parents = [None]
        self.path = None

    def plan(self):
        """执行RRT规划"""
        for i in range(self.max_iter):
            # 随机采样
            if np.random.random() < 0.1:  # 10%概率直接采样目标点
                random_point = self.goal
            else:
                random_point = self._random_sample()

            # 找到最近的节点
            nearest_idx = self._nearest_neighbor(random_point)
            nearest_node = self.nodes[nearest_idx]

            # 向随机点方向移动一步
            direction = random_point - nearest_node
            distance = np.linalg.norm(direction)
            if distance > 0:
                direction = direction / distance
                new_point = nearest_node + direction * min(self.step_size, distance)

                # 检查碰撞
                if not self._check_collision(nearest_node, new_point):
                    self.nodes.append(new_point)
                    self.parents.append(nearest_idx)

                    # 检查是否到达目标
                    if np.linalg.norm(new_point - self.goal) < self.step_size:
                        self._extract_path()
                        return True

        # 尝试连接最后一点到目标
        self._try_connect_to_goal()
        return self.path is not None

    def _random_sample(self):
        """在工作空间内随机采样"""
        return np.array([
            np.random.uniform(self.bounds[0], self.bounds[1]),
            np.random.uniform(self.bounds[2], self.bounds[3]),
            np.random.uniform(self.bounds[4], self.bounds[5])
        ])

    def _nearest_neighbor(self, point):
        """找到最近的节点"""
        distances = [np.linalg.norm(node - point) for node in self.nodes]
        return np.argmin(distances)

    def _check_collision(self, point1, point2):
        """检查线段是否与障碍物碰撞"""
        # 简化碰撞检测：检查中间点
        num_checks = max(2, int(np.linalg.norm(point2 - point1) / 0.05))
        for i in range(num_checks + 1):
            t = i / num_checks
            check_point = point1 + t * (point2 - point1)

            for obstacle in self.obstacles:
                if obstacle['type'] == 'sphere':
                    dist = np.linalg.norm(check_point - obstacle['center'])
                    if dist < obstacle['radius']:
                        return True
                elif obstacle['type'] == 'box':
                    # 检查点是否在立方体内
                    center = obstacle['center']
                    size = obstacle['size']
                    if (abs(check_point[0] - center[0]) <= size[0] and
                        abs(check_point[1] - center[1]) <= size[1] and
                        abs(check_point[2] - center[2]) <= size[2]):
                        return True
        return False

    def _extract_path(self):
        """从树中提取路径"""
        path = [self.goal]
        current_idx = len(self.nodes) - 1

        while current_idx is not None:
            path.append(self.nodes[current_idx])
            current_idx = self.parents[current_idx]

        self.path = list(reversed(path))

    def _try_connect_to_goal(self):
        """尝试连接最近的点到目标"""
        distances = [np.linalg.norm(node - self.goal) for node in self.nodes]
        nearest_idx = np.argmin(distances)

        if not self._check_collision(self.nodes[nearest_idx], self.goal):
            self.nodes.append(self.goal)
            self.parents.append(nearest_idx)
            self._extract_path()


class PRMPlanner:
    """概率路线图(PRM)路径规划算法"""

    def __init__(self, start, goal, obstacles, bounds, n_samples=1000, k_neighbors=10):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles
        self.bounds = bounds
        self.n_samples = n_samples
        self.k_neighbors = k_neighbors

        self.samples = []
        self.graph = nx.Graph()
        self.path = None

    def plan(self):
        """执行PRM规划"""
        # 1. 采样
        self._sample_configuration_space()

        # 2. 添加起点和终点
        self.samples.append(self.start)
        self.samples.append(self.goal)
        start_idx = len(self.samples) - 2
        goal_idx = len(self.samples) - 1

        # 3. 构建KD树用于最近邻搜索
        kdtree = cKDTree(self.samples)

        # 4. 构建图
        for i, sample in enumerate(self.samples):
            # 找到k个最近邻
            distances, indices = kdtree.query(sample, k=min(self.k_neighbors + 1, len(self.samples)))

            for j, neighbor_idx in enumerate(indices[1:]):  # 跳过自己
                neighbor = self.samples[neighbor_idx]

                # 检查碰撞
                if not self._check_collision(sample, neighbor):
                    # 计算距离作为边的权重
                    distance = np.linalg.norm(sample - neighbor)
                    self.graph.add_edge(i, neighbor_idx, weight=distance)

        # 5. 搜索路径
        try:
            path_indices = nx.shortest_path(self.graph, start_idx, goal_idx, weight='weight')
            self.path = [self.samples[i] for i in path_indices]
            return True
        except:
            return False

    def _sample_configuration_space(self):
        """在自由空间采样"""
        for _ in range(self.n_samples):
            while True:
                sample = np.array([
                    np.random.uniform(self.bounds[0], self.bounds[1]),
                    np.random.uniform(self.bounds[2], self.bounds[3]),
                    np.random.uniform(self.bounds[4], self.bounds[5])
                ])

                # 检查采样点是否在自由空间
                if not self._point_in_obstacle(sample):
                    self.samples.append(sample)
                    break

    def _point_in_obstacle(self, point):
        """检查点是否在障碍物内"""
        for obstacle in self.obstacles:
            if obstacle['type'] == 'sphere':
                dist = np.linalg.norm(point - obstacle['center'])
                if dist < obstacle['radius']:
                    return True
            elif obstacle['type'] == 'box':
                center = obstacle['center']
                size = obstacle['size']
                if (abs(point[0] - center[0]) <= size[0] and
                    abs(point[1] - center[1]) <= size[1] and
                    abs(point[2] - center[2]) <= size[2]):
                    return True
        return False

    def _check_collision(self, point1, point2):
        """检查线段是否与障碍物碰撞"""
        # 与RRT相同的碰撞检测
        num_checks = max(2, int(np.linalg.norm(point2 - point1) / 0.05))
        for i in range(num_checks + 1):
            t = i / num_checks
            check_point = point1 + t * (point2 - point1)

            for obstacle in self.obstacles:
                if obstacle['type'] == 'sphere':
                    dist = np.linalg.norm(check_point - obstacle['center'])
                    if dist < obstacle['radius']:
                        return True
                elif obstacle['type'] == 'box':
                    center = obstacle['center']
                    size = obstacle['size']
                    if (abs(check_point[0] - center[0]) <= size[0] and
                        abs(check_point[1] - center[1]) <= size[1] and
                        abs(check_point[2] - center[2]) <= size[2]):
                        return True
        return False


class AStarPlanner:
    """A*网格搜索路径规划算法"""

    def __init__(self, start, goal, obstacles, bounds, grid_size=0.1):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = obstacles
        self.bounds = bounds
        self.grid_size = grid_size

        # 创建网格
        x_grid = int((bounds[1] - bounds[0]) / grid_size)
        y_grid = int((bounds[3] - bounds[2]) / grid_size)
        z_grid = int((bounds[5] - bounds[4]) / grid_size)

        self.grid_shape = (x_grid, y_grid, z_grid)
        self.grid = np.zeros(self.grid_shape, dtype=bool)  # True表示障碍物

        self._initialize_grid()
        self.path = None

    def _initialize_grid(self):
        """初始化网格障碍物"""
        # 将障碍物标记在网格上
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                for k in range(self.grid_shape[2]):
                    world_pos = self._grid_to_world((i, j, k))

                    # 检查是否在障碍物内
                    if self._point_in_obstacle(world_pos):
                        self.grid[i, j, k] = True

    def _grid_to_world(self, grid_pos):
        """网格坐标转世界坐标"""
        return np.array([
            self.bounds[0] + grid_pos[0] * self.grid_size,
            self.bounds[2] + grid_pos[1] * self.grid_size,
            self.bounds[4] + grid_pos[2] * self.grid_size
        ])

    def _world_to_grid(self, world_pos):
        """世界坐标转网格坐标"""
        return (
            int((world_pos[0] - self.bounds[0]) / self.grid_size),
            int((world_pos[1] - self.bounds[2]) / self.grid_size),
            int((world_pos[2] - self.bounds[4]) / self.grid_size)
        )

    def _point_in_obstacle(self, point):
        """检查点是否在障碍物内"""
        for obstacle in self.obstacles:
            if obstacle['type'] == 'sphere':
                dist = np.linalg.norm(point - obstacle['center'])
                if dist < obstacle['radius']:
                    return True
            elif obstacle['type'] == 'box':
                center = obstacle['center']
                size = obstacle['size']
                if (abs(point[0] - center[0]) <= size[0] and
                    abs(point[1] - center[1]) <= size[1] and
                    abs(point[2] - center[2]) <= size[2]):
                    return True
        return False

    def plan(self):
        """执行A*规划"""
        start_grid = self._world_to_grid(self.start)
        goal_grid = self._world_to_grid(self.goal)

        # 检查起点和终点是否有效
        if (self._is_out_of_bounds(start_grid) or self._is_out_of_bounds(goal_grid) or
            self.grid[start_grid] or self.grid[goal_grid]):
            return False

        # A*算法
        open_set = []
        heapq.heappush(open_set, (0, start_grid))

        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self._heuristic(start_grid, goal_grid)}

        # 26个方向的邻居（3D Moore邻域）
        directions = [(dx, dy, dz) for dx in (-1, 0, 1)
                               for dy in (-1, 0, 1)
                               for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal_grid:
                self._reconstruct_path(came_from, current)
                return True

            for dx, dy, dz in directions:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)

                # 检查邻居是否有效
                if (self._is_out_of_bounds(neighbor) or self.grid[neighbor]):
                    continue

                # 计算移动成本（对角线成本较高）
                move_cost = np.sqrt(dx*dx + dy*dy + dz*dz)
                tentative_g_score = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return False

    def _is_out_of_bounds(self, grid_pos):
        """检查网格坐标是否越界"""
        return (grid_pos[0] < 0 or grid_pos[0] >= self.grid_shape[0] or
                grid_pos[1] < 0 or grid_pos[1] >= self.grid_shape[1] or
                grid_pos[2] < 0 or grid_pos[2] >= self.grid_shape[2])

    def _heuristic(self, a, b):
        """启发式函数（欧几里得距离）"""
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    def _reconstruct_path(self, came_from, current):
        """重建路径"""
        path_grid = [current]
        while current in came_from:
            current = came_from[current]
            path_grid.append(current)

        path_grid.reverse()
        self.path = [self._grid_to_world(pos) for pos in path_grid]


# ============================================================================
# 2. 轨迹优化算法
# ============================================================================

class TrajectoryOptimizer:
    """轨迹优化器"""

    def __init__(self, path):
        self.path = np.array(path)

    def smooth_path(self, weight_data=0.1, weight_smooth=0.3, tolerance=0.00001):
        """使用梯度下降平滑路径"""
        if len(self.path) < 3:
            return self.path

        smoothed = self.path.copy()

        change = tolerance
        while change >= tolerance:
            change = 0.0

            for i in range(1, len(self.path) - 1):
                for dim in range(3):
                    original = smoothed[i, dim]

                    # 数据项：保持接近原始点
                    data_term = weight_data * (self.path[i, dim] - smoothed[i, dim])

                    # 平滑项：保持相邻点接近
                    smooth_term = weight_smooth * (smoothed[i-1, dim] + smoothed[i+1, dim] - 2 * smoothed[i, dim])

                    smoothed[i, dim] += data_term + smooth_term
                    change += abs(original - smoothed[i, dim])

        return smoothed

    def interpolate_cubic_spline(self, num_points=100):
        """三次样条插值"""
        if len(self.path) < 4:
            # 如果点太少，使用线性插值
            t = np.linspace(0, 1, len(self.path))
            t_new = np.linspace(0, 1, num_points)

            spline_x = interp1d(t, self.path[:, 0], kind='cubic')
            spline_y = interp1d(t, self.path[:, 1], kind='cubic')
            spline_z = interp1d(t, self.path[:, 2], kind='cubic')

        else:
            # 参数化路径
            t = np.arange(len(self.path))
            t_new = np.linspace(0, len(self.path)-1, num_points)

            spline_x = interp1d(t, self.path[:, 0], kind='cubic')
            spline_y = interp1d(t, self.path[:, 1], kind='cubic')
            spline_z = interp1d(t, self.path[:, 2], kind='cubic')

        interpolated = np.column_stack([
            spline_x(t_new),
            spline_y(t_new),
            spline_z(t_new)
        ])

        return interpolated

    def minimize_jerk(self, total_time=5.0, dt=0.01):
        """最小化加加速度（jerk）轨迹"""
        n_points = len(self.path)
        t = np.linspace(0, total_time, n_points)

        # 创建五次多项式轨迹（最小化jerk）
        coefficients = []
        for dim in range(3):
            # 解决边界条件：位置、速度、加速度在起点和终点都为0
            A = np.zeros((6, 6))
            b = np.zeros(6)

            # 位置约束
            A[0, :] = [t[0]**i for i in range(6)]  # 起点位置
            b[0] = self.path[0, dim]
            A[1, :] = [t[-1]**i for i in range(6)]  # 终点位置
            b[1] = self.path[-1, dim]

            # 速度约束（起点和终点速度为0）
            A[2, :] = [i * t[0]**(i-1) if i > 0 else 0 for i in range(6)]  # 起点速度
            b[2] = 0
            A[3, :] = [i * t[-1]**(i-1) if i > 0 else 0 for i in range(6)]  # 终点速度
            b[3] = 0

            # 加速度约束（起点和终点加速度为0）
            A[4, :] = [i * (i-1) * t[0]**(i-2) if i > 1 else 0 for i in range(6)]  # 起点加速度
            b[4] = 0
            A[5, :] = [i * (i-1) * t[-1]**(i-2) if i > 1 else 0 for i in range(6)]  # 终点加速度
            b[5] = 0

            coeff = np.linalg.lstsq(A, b, rcond=None)[0]
            coefficients.append(coeff)

        # 生成轨迹
        time_points = np.arange(0, total_time, dt)
        n_steps = len(time_points)

        position = np.zeros((n_steps, 3))
        velocity = np.zeros((n_steps, 3))
        acceleration = np.zeros((n_steps, 3))
        jerk = np.zeros((n_steps, 3))

        for i, t in enumerate(time_points):
            for dim in range(3):
                coeff = coefficients[dim]
                # 位置
                position[i, dim] = sum(coeff[j] * t**j for j in range(6))
                # 速度
                velocity[i, dim] = sum(j * coeff[j] * t**(j-1) for j in range(1, 6))
                # 加速度
                acceleration[i, dim] = sum(j * (j-1) * coeff[j] * t**(j-2) for j in range(2, 6))
                # 加加速度
                jerk[i, dim] = sum(j * (j-1) * (j-2) * coeff[j] * t**(j-3) for j in range(3, 6))

        return position, velocity, acceleration, jerk, time_points

    def optimize_for_energy(self, total_time=5.0, dt=0.01):
        """能量优化轨迹"""
        # 简化的能量优化：最小化加速度平方和（与能量消耗相关）
        n_points = len(self.path)

        def cost_function(params):
            # params: 每个中间点的时间分配
            times = np.cumsum(np.concatenate([[0], params, [total_time - np.sum(params)]]))

            total_energy = 0
            for dim in range(3):
                # 计算样条插值
                tck, u = splprep([self.path[:, dim]], u=times, s=0)

                # 计算加速度
                der = splev(u, tck, der=2)

                # 能量近似为加速度平方的积分
                total_energy += np.trapz(der[0]**2, times)

            return total_energy

        # 初始时间分配（均匀分布）
        initial_params = np.ones(n_points-2) * (total_time / (n_points-1))

        # 边界条件：每段时间必须为正
        bounds = Bounds(0.01, total_time)

        # 约束：总时间不能超过total_time
        constraints = [{
            'type': 'ineq',
            'fun': lambda x: total_time - np.sum(x) - 0.01
        }]

        result = minimize(cost_function, initial_params,
                         bounds=bounds, constraints=constraints,
                         method='SLSQP')

        if result.success:
            optimal_times = np.cumsum(np.concatenate([[0], result.x, [total_time - np.sum(result.x)]]))

            # 使用优化后的时间生成轨迹
            trajectories = []
            for dim in range(3):
                tck, u = splprep([self.path[:, dim]], u=optimal_times, s=0)
                trajectories.append(tck)

            # 在更密集的时间点上采样
            time_points = np.linspace(0, total_time, int(total_time/dt))

            position = np.zeros((len(time_points), 3))
            velocity = np.zeros((len(time_points), 3))
            acceleration = np.zeros((len(time_points), 3))

            for i, t in enumerate(time_points):
                for dim in range(3):
                    pos = splev(t, trajectories[dim], der=0)
                    vel = splev(t, trajectories[dim], der=1)
                    acc = splev(t, trajectories[dim], der=2)

                    position[i, dim] = pos[0]
                    velocity[i, dim] = vel[0]
                    acceleration[i, dim] = acc[0]

            return position, velocity, acceleration, time_points

        else:
            # 优化失败，返回最小jerk轨迹
            return self.minimize_jerk(total_time, dt)[:4]


# ============================================================================
# 3. 逆运动学求解器（简化版本）
# ============================================================================

class SimplifiedInverseKinematics:
    """简化逆运动学求解器（适用于6自由度机械臂）"""

    def __init__(self, link_lengths=[0.35, 0.3, 0.15, 0.12, 0.05]):
        # 连杆长度：[上臂, 前臂, 腕部1, 腕部2, 末端]
        self.link_lengths = link_lengths
        self.total_length = sum(link_lengths)

    def solve(self, target_position, target_orientation=None):
        """求解逆运动学（简化几何方法）"""
        x, y, z = target_position

        # 1. 计算基座旋转（关节1）
        theta1 = np.arctan2(y, x)

        # 2. 计算距离和高度
        distance_xy = np.sqrt(x**2 + y**2)
        height = z

        # 3. 考虑机械臂的基座高度
        base_height = 0.1
        effective_height = height - base_height

        # 4. 简化逆运动学（平面2连杆）
        L1 = self.link_lengths[0]  # 上臂
        L2 = self.link_lengths[1]  # 前臂

        # 计算到目标的距离
        D = np.sqrt(distance_xy**2 + effective_height**2)

        # 检查是否可达
        if D > L1 + L2 or D < abs(L1 - L2):
            # 不可达，返回最接近的配置
            theta2 = np.pi/2  # 默认垂直
            theta3 = 0
        else:
            # 计算关节角度
            # 肘部角度（关节3）
            cos_theta3 = (D**2 - L1**2 - L2**2) / (2 * L1 * L2)
            cos_theta3 = np.clip(cos_theta3, -1, 1)
            theta3 = np.arccos(cos_theta3)

            # 肩部角度（关节2）
            alpha = np.arctan2(effective_height, distance_xy)
            beta = np.arcsin(L2 * np.sin(theta3) / D)
            theta2 = alpha - beta

        # 5. 腕部角度（简化）
        theta4 = 0  # 腕部旋转
        theta5 = np.pi/2  # 腕部俯仰（默认垂直向下）
        theta6 = 0  # 腕部横滚

        # 限制关节角度范围
        joint_angles = np.array([theta1, theta2, theta3, theta4, theta5, theta6])

        # 应用关节限制（根据您的模型）
        limits = [
            (-np.pi, np.pi),    # 关节1
            (-2.0, 2.0),        # 关节2
            (-2.5, 0.5),        # 关节3
            (-2.0, 2.0),        # 关节4
            (-1.8, 1.8),        # 关节5
            (-1.8, 1.8)         # 关节6
        ]

        for i in range(6):
            joint_angles[i] = np.clip(joint_angles[i], limits[i][0], limits[i][1])

        return joint_angles

    def forward_kinematics(self, joint_angles):
        """正运动学计算末端位置"""
        theta1, theta2, theta3, theta4, theta5, theta6 = joint_angles

        # 简化正运动学（不考虑所有连杆偏移）
        L1 = self.link_lengths[0]
        L2 = self.link_lengths[1]
        L3 = self.link_lengths[2]
        L4 = self.link_lengths[3]
        L5 = self.link_lengths[4]

        # 计算位置
        x = (L1 * np.sin(theta2) + L2 * np.sin(theta2 + theta3) +
             L3 * np.sin(theta2 + theta3) +
             L4 * np.sin(theta2 + theta3) +
             L5 * np.sin(theta2 + theta3)) * np.cos(theta1)

        y = (L1 * np.sin(theta2) + L2 * np.sin(theta2 + theta3) +
             L3 * np.sin(theta2 + theta3) +
             L4 * np.sin(theta2 + theta3) +
             L5 * np.sin(theta2 + theta3)) * np.sin(theta1)

        z = 0.1 + (L1 * np.cos(theta2) + L2 * np.cos(theta2 + theta3) +
                   L3 * np.cos(theta2 + theta3) +
                   L4 * np.cos(theta2 + theta3) +
                   L5 * np.cos(theta2 + theta3))

        return np.array([x, y, z])


# ============================================================================
# 4. 可视化系统
# ============================================================================

class TrajectoryPlanningVisualizer:
    """轨迹规划可视化系统"""

    def __init__(self):
        self.fig = plt.figure(figsize=(20, 12))
        self.setup_visualization()

    def setup_visualization(self):
        """设置可视化面板"""
        gs = self.fig.add_gridspec(3, 4)

        # 1. 3D工作空间与轨迹
        self.ax_3d = self.fig.add_subplot(gs[:2, :2], projection='3d')
        self.ax_3d.set_title('工作空间与轨迹规划', fontsize=14, fontweight='bold')
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.grid(True, alpha=0.3)

        # 2. XY平面投影
        self.ax_xy = self.fig.add_subplot(gs[0, 2])
        self.ax_xy.set_title('XY平面投影', fontsize=12, fontweight='bold')
        self.ax_xy.set_xlabel('X (m)')
        self.ax_xy.set_ylabel('Y (m)')
        self.ax_xy.grid(True, alpha=0.3)

        # 3. XZ平面投影
        self.ax_xz = self.fig.add_subplot(gs[0, 3])
        self.ax_xz.set_title('XZ平面投影', fontsize=12, fontweight='bold')
        self.ax_xz.set_xlabel('X (m)')
        self.ax_xz.set_ylabel('Z (m)')
        self.ax_xz.grid(True, alpha=0.3)

        # 4. YZ平面投影
        self.ax_yz = self.fig.add_subplot(gs[1, 2])
        self.ax_yz.set_title('YZ平面投影', fontsize=12, fontweight='bold')
        self.ax_yz.set_xlabel('Y (m)')
        self.ax_yz.set_ylabel('Z (m)')
        self.ax_yz.grid(True, alpha=0.3)

        # 5. 关节角度轨迹
        self.ax_joints = self.fig.add_subplot(gs[1, 3])
        self.ax_joints.set_title('关节角度轨迹', fontsize=12, fontweight='bold')
        self.ax_joints.set_xlabel('时间步')
        self.ax_joints.set_ylabel('角度 (rad)')
        self.ax_joints.grid(True, alpha=0.3)

        # 6. 速度曲线
        self.ax_velocity = self.fig.add_subplot(gs[2, 0])
        self.ax_velocity.set_title('关节速度', fontsize=12, fontweight='bold')
        self.ax_velocity.set_xlabel('时间步')
        self.ax_velocity.set_ylabel('速度 (rad/s)')
        self.ax_velocity.grid(True, alpha=0.3)

        # 7. 加速度曲线
        self.ax_acceleration = self.fig.add_subplot(gs[2, 1])
        self.ax_acceleration.set_title('关节加速度', fontsize=12, fontweight='bold')
        self.ax_acceleration.set_xlabel('时间步')
        self.ax_acceleration.set_ylabel('加速度 (rad/s²)')
        self.ax_acceleration.grid(True, alpha=0.3)

        # 8. 能量消耗
        self.ax_energy = self.fig.add_subplot(gs[2, 2])
        self.ax_energy.set_title('能量消耗', fontsize=12, fontweight='bold')
        self.ax_energy.set_xlabel('时间步')
        self.ax_energy.set_ylabel('能量 (J)')
        self.ax_energy.grid(True, alpha=0.3)

        # 9. 算法性能比较
        self.ax_comparison = self.fig.add_subplot(gs[2, 3])
        self.ax_comparison.set_title('算法性能比较', fontsize=12, fontweight='bold')
        self.ax_comparison.set_xlabel('算法')
        self.ax_comparison.set_ylabel('性能指标')
        self.ax_comparison.grid(True, alpha=0.3)

        plt.tight_layout()

    def plot_3d_workspace(self, obstacles, start, goal, path=None,
                         tree_nodes=None, tree_edges=None, show_obstacles=True):
        """绘制3D工作空间"""
        self.ax_3d.clear()

        if show_obstacles:
            # 绘制障碍物
            for obstacle in obstacles:
                if obstacle['type'] == 'sphere':
                    self._draw_sphere_3d(obstacle['center'], obstacle['radius'],
                                        color='red', alpha=0.3)
                elif obstacle['type'] == 'box':
                    self._draw_box_3d(obstacle['center'], obstacle['size'],
                                     color='red', alpha=0.3)

        # 绘制起点和终点
        self.ax_3d.scatter(start[0], start[1], start[2],
                          color='green', s=200, marker='o', label='起点')
        self.ax_3d.scatter(goal[0], goal[1], goal[2],
                          color='red', s=200, marker='*', label='终点')

        # 绘制RRT树（如果提供）
        if tree_nodes is not None:
            nodes_array = np.array(tree_nodes)
            self.ax_3d.scatter(nodes_array[:, 0], nodes_array[:, 1], nodes_array[:, 2],
                              color='lightblue', s=10, alpha=0.5, label='RRT节点')

            if tree_edges is not None:
                for edge in tree_edges:
                    points = np.array(edge)
                    self.ax_3d.plot(points[:, 0], points[:, 1], points[:, 2],
                                   'gray', alpha=0.3, linewidth=0.5)

        # 绘制路径
        if path is not None:
            path_array = np.array(path)
            self.ax_3d.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2],
                           'orange', linewidth=3, label='路径')
            self.ax_3d.scatter(path_array[:, 0], path_array[:, 1], path_array[:, 2],
                              color='orange', s=30, alpha=0.7)

        # 设置坐标轴范围
        all_points = []
        if path is not None:
            all_points.extend(path)
        if tree_nodes is not None:
            all_points.extend(tree_nodes)

        if all_points:
            all_points = np.array(all_points)
            min_coords = all_points.min(axis=0) - 0.1
            max_coords = all_points.max(axis=0) + 0.1

            self.ax_3d.set_xlim(min_coords[0], max_coords[0])
            self.ax_3d.set_ylim(min_coords[1], max_coords[1])
            self.ax_3d.set_zlim(min_coords[2], max_coords[2])

        self.ax_3d.legend(loc='upper right')
        self.ax_3d.set_title('工作空间与轨迹规划')
        self.ax_3d.grid(True, alpha=0.3)

    def plot_projection_views(self, path, obstacles=None):
        """绘制投影视图"""
        if path is None or len(path) == 0:
            return

        path_array = np.array(path)

        # XY投影
        self.ax_xy.clear()
        self.ax_xy.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=2)
        self.ax_xy.scatter(path_array[:, 0], path_array[:, 1], c=range(len(path_array)),
                          cmap='viridis', s=30, alpha=0.7)
        self.ax_xy.scatter(path_array[0, 0], path_array[0, 1], color='green', s=100, marker='o')
        self.ax_xy.scatter(path_array[-1, 0], path_array[-1, 1], color='red', s=100, marker='*')
        self.ax_xy.set_title('XY平面投影')
        self.ax_xy.grid(True, alpha=0.3)
        self.ax_xy.axis('equal')

        # XZ投影
        self.ax_xz.clear()
        self.ax_xz.plot(path_array[:, 0], path_array[:, 2], 'g-', linewidth=2)
        self.ax_xz.scatter(path_array[:, 0], path_array[:, 2], c=range(len(path_array)),
                          cmap='viridis', s=30, alpha=0.7)
        self.ax_xz.scatter(path_array[0, 0], path_array[0, 2], color='green', s=100, marker='o')
        self.ax_xz.scatter(path_array[-1, 0], path_array[-1, 2], color='red', s=100, marker='*')
        self.ax_xz.set_title('XZ平面投影')
        self.ax_xz.grid(True, alpha=0.3)
        self.ax_xz.axis('equal')

        # YZ投影
        self.ax_yz.clear()
        self.ax_yz.plot(path_array[:, 1], path_array[:, 2], 'r-', linewidth=2)
        self.ax_yz.scatter(path_array[:, 1], path_array[:, 2], c=range(len(path_array)),
                          cmap='viridis', s=30, alpha=0.7)
        self.ax_yz.scatter(path_array[0, 1], path_array[0, 2], color='green', s=100, marker='o')
        self.ax_yz.scatter(path_array[-1, 1], path_array[-1, 2], color='red', s=100, marker='*')
        self.ax_yz.set_title('YZ平面投影')
        self.ax_yz.grid(True, alpha=0.3)
        self.ax_yz.axis('equal')

    def plot_joint_trajectories(self, joint_trajectory, time_points=None):
        """绘制关节角度轨迹"""
        self.ax_joints.clear()

        if joint_trajectory is None or len(joint_trajectory) == 0:
            return

        joint_array = np.array(joint_trajectory)
        n_joints = joint_array.shape[1]

        if time_points is None:
            time_points = range(len(joint_trajectory))

        colors = plt.cm.tab10(np.linspace(0, 1, n_joints))

        for i in range(n_joints):
            self.ax_joints.plot(time_points, joint_array[:, i],
                               color=colors[i], linewidth=2, label=f'关节 {i+1}')

        self.ax_joints.legend(loc='upper right')
        self.ax_joints.set_title('关节角度轨迹')
        self.ax_joints.set_xlabel('时间步')
        self.ax_joints.set_ylabel('角度 (rad)')
        self.ax_joints.grid(True, alpha=0.3)

    def plot_velocity_acceleration(self, velocity, acceleration, time_points=None):
        """绘制速度和加速度曲线"""
        if velocity is None or acceleration is None:
            return

        vel_array = np.array(velocity)
        acc_array = np.array(acceleration)
        n_joints = vel_array.shape[1]

        if time_points is None:
            time_points = range(len(velocity))

        colors = plt.cm.tab10(np.linspace(0, 1, n_joints))

        # 速度曲线
        self.ax_velocity.clear()
        for i in range(n_joints):
            self.ax_velocity.plot(time_points, vel_array[:, i],
                                 color=colors[i], linewidth=1.5, alpha=0.7, label=f'关节 {i+1}')
        self.ax_velocity.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        self.ax_velocity.set_title('关节速度')
        self.ax_velocity.set_xlabel('时间步')
        self.ax_velocity.set_ylabel('速度 (rad/s)')
        self.ax_velocity.legend(loc='upper right', fontsize=8)
        self.ax_velocity.grid(True, alpha=0.3)

        # 加速度曲线
        self.ax_acceleration.clear()
        for i in range(n_joints):
            self.ax_acceleration.plot(time_points, acc_array[:, i],
                                     color=colors[i], linewidth=1.5, alpha=0.7, label=f'关节 {i+1}')
        self.ax_acceleration.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        self.ax_acceleration.set_title('关节加速度')
        self.ax_acceleration.set_xlabel('时间步')
        self.ax_acceleration.set_ylabel('加速度 (rad/s²)')
        self.ax_acceleration.legend(loc='upper right', fontsize=8)
        self.ax_acceleration.grid(True, alpha=0.3)

    def plot_energy_consumption(self, velocity, acceleration, torque=None, time_points=None):
        """绘制能量消耗曲线"""
        if velocity is None or acceleration is None:
            return

        vel_array = np.array(velocity)
        acc_array = np.array(acceleration)

        if time_points is None:
            time_points = range(len(velocity))

        # 计算动能（假设所有连杆质量相同）
        kinetic_energy = 0.5 * np.sum(vel_array**2, axis=1)

        # 计算功率（速度 × 加速度的绝对值，简化模型）
        if torque is not None:
            torque_array = np.array(torque)
            power = np.abs(vel_array * torque_array)
        else:
            # 简化的功率计算
            power = np.abs(vel_array * acc_array)

        total_power = np.sum(power, axis=1)

        # 积分得到能量
        if len(time_points) > 1:
            dt = time_points[1] - time_points[0] if len(time_points) > 1 else 0.01
        else:
            dt = 0.01

        energy = np.cumsum(total_power) * dt

        self.ax_energy.clear()

        # 绘制能量曲线
        self.ax_energy.plot(time_points, energy, 'b-', linewidth=2, label='累计能量')
        self.ax_energy.fill_between(time_points, 0, energy, alpha=0.3, color='blue')

        # 绘制功率曲线（次坐标轴）
        ax2 = self.ax_energy.twinx()
        ax2.plot(time_points, total_power, 'r--', linewidth=1.5, alpha=0.7, label='瞬时功率')
        ax2.set_ylabel('功率 (W)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # 添加图例
        lines1, labels1 = self.ax_energy.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        self.ax_energy.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        self.ax_energy.set_title(f'能量消耗: {energy[-1]:.3f} J')
        self.ax_energy.set_xlabel('时间步')
        self.ax_energy.set_ylabel('能量 (J)', color='blue')
        self.ax_energy.tick_params(axis='y', labelcolor='blue')
        self.ax_energy.grid(True, alpha=0.3)

    def plot_algorithm_comparison(self, algorithm_results):
        """绘制算法性能比较"""
        self.ax_comparison.clear()

        algorithms = list(algorithm_results.keys())
        metrics = ['计算时间', '路径长度', '平滑度', '能量消耗']

        # 归一化数据
        normalized_data = {}
        for metric in metrics:
            values = [algorithm_results[algo].get(metric, 0) for algo in algorithms]
            if max(values) > 0:
                normalized = [v / max(values) for v in values]
            else:
                normalized = values
            normalized_data[metric] = normalized

        # 绘制雷达图
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形

        for i, algo in enumerate(algorithms):
            values = [normalized_data[metric][i] for metric in metrics]
            values += values[:1]  # 闭合图形

            self.ax_comparison.plot(angles, values, 'o-', linewidth=2,
                                   label=algo, markersize=8)

        # 设置角度标签
        self.ax_comparison.set_xticks(angles[:-1])
        self.ax_comparison.set_xticklabels(metrics)
        self.ax_comparison.set_ylim(0, 1)
        self.ax_comparison.set_title('算法性能比较雷达图')
        self.ax_comparison.legend(loc='upper right')
        self.ax_comparison.grid(True, alpha=0.3)

    def _draw_sphere_3d(self, center, radius, color='red', alpha=0.3):
        """绘制3D球体"""
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)

        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

        self.ax_3d.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

    def _draw_box_3d(self, center, size, color='red', alpha=0.3):
        """绘制3D立方体"""
        # 定义立方体的8个顶点
        x = center[0]
        y = center[1]
        z = center[2]
        sx = size[0]
        sy = size[1]
        sz = size[2]

        vertices = np.array([
            [x-sx, y-sy, z-sz], [x+sx, y-sy, z-sz],
            [x+sx, y+sy, z-sz], [x-sx, y+sy, z-sz],
            [x-sx, y-sy, z+sz], [x+sx, y-sy, z+sz],
            [x+sx, y+sy, z+sz], [x-sx, y+sy, z+sz]
        ])

        # 定义立方体的6个面
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
            [vertices[0], vertices[3], vertices[7], vertices[4]]   # 左面
        ]

        # 绘制立方体
        poly3d = Poly3DCollection(faces, alpha=alpha, linewidths=1, edgecolors='darkred')
        poly3d.set_facecolor(color)
        self.ax_3d.add_collection3d(poly3d)

    def save_plot(self, filename):
        """保存图表"""
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {filename}")


# ============================================================================
# 5. 主程序 - 轨迹规划演示系统
# ============================================================================

class TrajectoryPlanningDemo:
    """轨迹规划演示系统"""

    def __init__(self):
        self.visualizer = TrajectoryPlanningVisualizer()

        # 定义工作空间边界
        self.bounds = [-0.8, 0.8, -0.8, 0.8, 0.0, 1.0]  # [xmin, xmax, ymin, ymax, zmin, zmax]

        # 定义障碍物
        self.obstacles = [
            {'type': 'sphere', 'center': [0.3, 0.3, 0.3], 'radius': 0.15},
            {'type': 'sphere', 'center': [-0.4, -0.3, 0.4], 'radius': 0.12},
            {'type': 'box', 'center': [0.0, 0.0, 0.6], 'size': [0.2, 0.2, 0.1]},
            {'type': 'box', 'center': [0.5, -0.5, 0.2], 'size': [0.15, 0.15, 0.3]}
        ]

        # 定义起点和终点
        self.start = np.array([-0.6, -0.6, 0.1])
        self.goal = np.array([0.6, 0.6, 0.8])

        # 逆运动学求解器
        self.ik_solver = SimplifiedInverseKinematics()

        # 存储结果
        self.results = {}

    def run_all_algorithms(self):
        """运行所有规划算法"""
        print("=" * 60)
        print("机械臂轨迹规划与优化演示系统")
        print("=" * 60)

        algorithms = [
            ('RRT', self.run_rrt),
            ('PRM', self.run_prm),
            ('A*', self.run_astar)
        ]

        for algo_name, algo_func in algorithms:
            print(f"\n运行{algo_name}算法...")
            success = algo_func()

            if success:
                print(f"  ✓ {algo_name}规划成功")
            else:
                print(f"  ✗ {algo_name}规划失败")

        # 比较算法性能
        self.compare_algorithms()

        # 显示可视化
        plt.show()

    def run_rrt(self):
        """运行RRT算法"""
        start_time = time.time()

        # 创建RRT规划器
        rrt = RRTPlanner(self.start, self.goal, self.obstacles,
                        self.bounds, step_size=0.1, max_iter=3000)

        # 执行规划
        success = rrt.plan()

        if success:
            # 优化轨迹
            optimizer = TrajectoryOptimizer(rrt.path)
            smoothed_path = optimizer.smooth_path()
            interpolated_path = optimizer.interpolate_cubic_spline(num_points=100)

            # 计算逆运动学
            joint_trajectory = []
            for point in interpolated_path:
                joint_angles = self.ik_solver.solve(point)
                joint_trajectory.append(joint_angles)

            # 计算最小jerk轨迹
            position, velocity, acceleration, jerk, time_points = optimizer.minimize_jerk()

            # 存储结果
            self.results['RRT'] = {
                'path': rrt.path,
                'smoothed_path': smoothed_path,
                'interpolated_path': interpolated_path,
                'joint_trajectory': joint_trajectory,
                'velocity': velocity,
                'acceleration': acceleration,
                'jerk': jerk,
                'time_points': time_points,
                'computation_time': time.time() - start_time,
                'path_length': self._calculate_path_length(interpolated_path),
                'smoothness': self._calculate_smoothness(interpolated_path)
            }

            # 可视化
            self.visualizer.plot_3d_workspace(
                self.obstacles, self.start, self.goal,
                path=interpolated_path,
                tree_nodes=rrt.nodes,
                tree_edges=None
            )

            self.visualizer.plot_projection_views(interpolated_path)

            if len(joint_trajectory) > 0:
                self.visualizer.plot_joint_trajectories(joint_trajectory)
                self.visualizer.plot_velocity_acceleration(velocity, acceleration)
                self.visualizer.plot_energy_consumption(velocity, acceleration)

            return True

        return False

    def run_prm(self):
        """运行PRM算法"""
        start_time = time.time()

        # 创建PRM规划器
        prm = PRMPlanner(self.start, self.goal, self.obstacles,
                        self.bounds, n_samples=500, k_neighbors=10)

        # 执行规划
        success = prm.plan()

        if success:
            # 优化轨迹
            optimizer = TrajectoryOptimizer(prm.path)
            smoothed_path = optimizer.smooth_path()
            interpolated_path = optimizer.interpolate_cubic_spline(num_points=100)

            # 计算逆运动学
            joint_trajectory = []
            for point in interpolated_path:
                joint_angles = self.ik_solver.solve(point)
                joint_trajectory.append(joint_angles)

            # 计算最小jerk轨迹
            position, velocity, acceleration, jerk, time_points = optimizer.minimize_jerk()

            # 存储结果
            self.results['PRM'] = {
                'path': prm.path,
                'smoothed_path': smoothed_path,
                'interpolated_path': interpolated_path,
                'joint_trajectory': joint_trajectory,
                'velocity': velocity,
                'acceleration': acceleration,
                'jerk': jerk,
                'time_points': time_points,
                'computation_time': time.time() - start_time,
                'path_length': self._calculate_path_length(interpolated_path),
                'smoothness': self._calculate_smoothness(interpolated_path)
            }

            # 可视化
            self.visualizer.plot_3d_workspace(
                self.obstacles, self.start, self.goal,
                path=interpolated_path
            )

            self.visualizer.plot_projection_views(interpolated_path)

            if len(joint_trajectory) > 0:
                self.visualizer.plot_joint_trajectories(joint_trajectory)
                self.visualizer.plot_velocity_acceleration(velocity, acceleration)
                self.visualizer.plot_energy_consumption(velocity, acceleration)

            return True

        return False

    def run_astar(self):
        """运行A*算法"""
        start_time = time.time()

        # 创建A*规划器
        astar = AStarPlanner(self.start, self.goal, self.obstacles,
                            self.bounds, grid_size=0.05)

        # 执行规划
        success = astar.plan()

        if success:
            # 优化轨迹
            optimizer = TrajectoryOptimizer(astar.path)
            smoothed_path = optimizer.smooth_path()
            interpolated_path = optimizer.interpolate_cubic_spline(num_points=100)

            # 计算逆运动学
            joint_trajectory = []
            for point in interpolated_path:
                joint_angles = self.ik_solver.solve(point)
                joint_trajectory.append(joint_angles)

            # 计算最小jerk轨迹
            position, velocity, acceleration, jerk, time_points = optimizer.minimize_jerk()

            # 存储结果
            self.results['A*'] = {
                'path': astar.path,
                'smoothed_path': smoothed_path,
                'interpolated_path': interpolated_path,
                'joint_trajectory': joint_trajectory,
                'velocity': velocity,
                'acceleration': acceleration,
                'jerk': jerk,
                'time_points': time_points,
                'computation_time': time.time() - start_time,
                'path_length': self._calculate_path_length(interpolated_path),
                'smoothness': self._calculate_smoothness(interpolated_path)
            }

            # 可视化
            self.visualizer.plot_3d_workspace(
                self.obstacles, self.start, self.goal,
                path=interpolated_path
            )

            self.visualizer.plot_projection_views(interpolated_path)

            if len(joint_trajectory) > 0:
                self.visualizer.plot_joint_trajectories(joint_trajectory)
                self.visualizer.plot_velocity_acceleration(velocity, acceleration)
                self.visualizer.plot_energy_consumption(velocity, acceleration)

            return True

        return False

    def compare_algorithms(self):
        """比较算法性能"""
        if not self.results:
            return

        # 准备比较数据
        comparison_data = {}
        for algo_name, result in self.results.items():
            comparison_data[algo_name] = {
                '计算时间': result['computation_time'],
                '路径长度': result['path_length'],
                '平滑度': result['smoothness'],
                '能量消耗': np.sum(result['acceleration']**2)  # 简化的能量指标
            }

        # 绘制比较图
        self.visualizer.plot_algorithm_comparison(comparison_data)

        # 打印比较结果
        print("\n" + "=" * 60)
        print("算法性能比较")
        print("=" * 60)

        print(f"{'算法':<10} {'计算时间(s)':<12} {'路径长度(m)':<12} {'平滑度':<12}")
        print("-" * 50)

        for algo_name, result in self.results.items():
            print(f"{algo_name:<10} {result['computation_time']:<12.4f} "
                  f"{result['path_length']:<12.4f} {result['smoothness']:<12.4f}")

        # 保存结果
        self.save_results()

    def _calculate_path_length(self, path):
        """计算路径长度"""
        if len(path) < 2:
            return 0

        total_length = 0
        for i in range(1, len(path)):
            total_length += np.linalg.norm(path[i] - path[i-1])

        return total_length

    def _calculate_smoothness(self, path):
        """计算路径平滑度（角度变化总和）"""
        if len(path) < 3:
            return 0

        total_angle_change = 0
        for i in range(1, len(path) - 1):
            v1 = path[i] - path[i-1]
            v2 = path[i+1] - path[i]

            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                total_angle_change += angle

        # 平滑度定义为角度变化的倒数
        smoothness = 1.0 / (total_angle_change + 1e-6)
        return smoothness

    def save_results(self):
        """保存规划结果"""
        # 保存数据
        with open('trajectory_planning_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)

        # 保存可视化图表
        self.visualizer.save_plot('trajectory_planning_comparison.png')

        # 生成报告
        report = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'start_position': self.start.tolist(),
            'goal_position': self.goal.tolist(),
            'obstacles': self.obstacles,
            'bounds': self.bounds,
            'algorithm_results': {}
        }

        for algo_name, result in self.results.items():
            report['algorithm_results'][algo_name] = {
                'computation_time': result['computation_time'],
                'path_length': result['path_length'],
                'smoothness': result['smoothness'],
                'num_path_points': len(result['path'])
            }

        with open('trajectory_planning_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n结果已保存:")
        print(f"  数据文件: trajectory_planning_results.pkl")
        print(f"  图表文件: trajectory_planning_comparison.png")
        print(f"  报告文件: trajectory_planning_report.json")


# ============================================================================
# 6. 交互式演示和动画
# ============================================================================

class InteractiveTrajectoryDemo:
    """交互式轨迹演示（包含动画）"""

    def __init__(self):
        self.fig = plt.figure(figsize=(15, 10))
        self.demo = TrajectoryPlanningDemo()

    def create_animation(self, algorithm='RRT'):
        """创建轨迹执行动画"""
        if algorithm not in self.demo.results:
            print(f"算法 {algorithm} 的结果不存在")
            return

        result = self.demo.results[algorithm]
        trajectory = result['interpolated_path']
        joint_trajectory = result['joint_trajectory']

        # 创建子图
        ax1 = self.fig.add_subplot(2, 2, 1, projection='3d')
        ax2 = self.fig.add_subplot(2, 2, 2)
        ax3 = self.fig.add_subplot(2, 2, 3)
        ax4 = self.fig.add_subplot(2, 2, 4)

        # 初始化图表
        ax1.set_title(f'{algorithm} 轨迹动画')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.set_zlim(0, 1)

        # 绘制障碍物
        for obstacle in self.demo.obstacles:
            if obstacle['type'] == 'sphere':
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x = obstacle['radius'] * np.outer(np.cos(u), np.sin(v)) + obstacle['center'][0]
                y = obstacle['radius'] * np.outer(np.sin(u), np.sin(v)) + obstacle['center'][1]
                z = obstacle['radius'] * np.outer(np.ones(np.size(u)), np.cos(v)) + obstacle['center'][2]
                ax1.plot_surface(x, y, z, color='red', alpha=0.3)

        # 绘制完整轨迹
        traj_line, = ax1.plot([], [], [], 'b-', alpha=0.3)
        traj_points = ax1.scatter([], [], [], c='blue', s=10, alpha=0.5)

        # 当前点
        current_point, = ax1.plot([], [], [], 'ro', markersize=10)

        # 关节角度图
        joint_lines = []
        colors = plt.cm.tab10(np.linspace(0, 1, 6))
        for i in range(6):
            line, = ax2.plot([], [], color=colors[i], label=f'关节{i+1}')
            joint_lines.append(line)

        ax2.set_title('关节角度')
        ax2.set_xlabel('时间步')
        ax2.set_ylabel('角度 (rad)')
        ax2.legend()
        ax2.set_xlim(0, len(joint_trajectory))
        ax2.set_ylim(-3.5, 3.5)

        # 速度图
        velocity_lines = []
        for i in range(6):
            line, = ax3.plot([], [], color=colors[i])
            velocity_lines.append(line)

        ax3.set_title('关节速度')
        ax3.set_xlabel('时间步')
        ax3.set_ylabel('速度 (rad/s)')
        ax3.set_xlim(0, len(joint_trajectory))

        # 能量图
        energy_line, = ax4.plot([], [], 'b-')
        ax4.set_title('能量消耗')
        ax4.set_xlabel('时间步')
        ax4.set_ylabel('能量 (J)')
        ax4.set_xlim(0, len(joint_trajectory))

        plt.tight_layout()

        # 动画更新函数
        def update(frame):
            # 更新3D轨迹
            traj_line.set_data(trajectory[:frame, 0], trajectory[:frame, 1])
            traj_line.set_3d_properties(trajectory[:frame, 2])

            traj_points._offsets3d = (trajectory[:frame, 0],
                                     trajectory[:frame, 1],
                                     trajectory[:frame, 2])

            current_point.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])
            current_point.set_3d_properties([trajectory[frame, 2]])

            # 更新关节角度
            for i in range(6):
                joint_lines[i].set_data(range(frame+1),
                                       [jt[i] for jt in joint_trajectory[:frame+1]])

            # 更新速度（简化计算）
            if frame > 0:
                velocities = []
                for i in range(6):
                    if frame < len(joint_trajectory):
                        vel = (joint_trajectory[frame][i] - joint_trajectory[frame-1][i]) / 0.01
                        velocities.append(vel)
                    else:
                        velocities.append(0)

                for i in range(6):
                    if frame < len(velocity_lines[i].get_xdata()):
                        # 更新现有数据
                        xdata = list(velocity_lines[i].get_xdata())
                        ydata = list(velocity_lines[i].get_ydata())
                        xdata.append(frame)
                        ydata.append(velocities[i])
                        velocity_lines[i].set_data(xdata, ydata)
                    else:
                        velocity_lines[i].set_data(range(frame+1), velocities[:frame+1])

            # 更新能量（简化计算）
            if frame > 0:
                energy = 0
                for i in range(frame):
                    # 计算动能
                    kinetic = 0
                    for j in range(6):
                        if i > 0:
                            vel = (joint_trajectory[i][j] - joint_trajectory[i-1][j]) / 0.01
                            kinetic += 0.5 * 0.1 * vel**2  # 假设质量为0.1kg

                    energy += kinetic

                energy_line.set_data(range(frame+1),
                                   [self._calculate_energy(joint_trajectory[:k+1])
                                    for k in range(frame+1)])

            return (traj_line, traj_points, current_point, *joint_lines,
                    *velocity_lines, energy_line)

        # 创建动画
        ani = animation.FuncAnimation(
            self.fig, update, frames=len(trajectory),
            interval=50, blit=False, repeat=True
        )

        # 保存动画
        try:
            ani.save(f'trajectory_animation_{algorithm}.gif', writer='pillow', fps=20)
            print(f"动画已保存: trajectory_animation_{algorithm}.gif")
        except:
            print("警告: 无法保存GIF动画，显示实时动画")

        plt.show()
        return ani

    def _calculate_energy(self, joint_trajectory):
        """计算能量消耗（简化）"""
        if len(joint_trajectory) < 2:
            return 0

        energy = 0
        for i in range(1, len(joint_trajectory)):
            # 计算每个关节的速度平方和
            for j in range(6):
                vel = (joint_trajectory[i][j] - joint_trajectory[i-1][j]) / 0.01
                energy += 0.5 * 0.1 * vel**2  # 假设质量为0.1kg

        return energy


# ============================================================================
# 7. 主入口
# ============================================================================

def main():
    """主函数"""
    print("=" * 60)
    print("机械臂轨迹规划与优化可视化系统")
    print("=" * 60)

    # 选项菜单
    print("\n请选择运行模式:")
    print("1. 运行所有算法并比较")
    print("2. 运行单个算法 (RRT)")
    print("3. 运行单个算法 (PRM)")
    print("4. 运行单个算法 (A*)")
    print("5. 创建轨迹动画")
    print("6. 退出")

    choice = input("\n请输入选项 (1-6): ")

    if choice == '1':
        # 运行所有算法
        demo = TrajectoryPlanningDemo()
        demo.run_all_algorithms()

    elif choice == '2':
        # 运行RRT算法
        demo = TrajectoryPlanningDemo()
        demo.run_rrt()
        plt.show()

    elif choice == '3':
        # 运行PRM算法
        demo = TrajectoryPlanningDemo()
        demo.run_prm()
        plt.show()

    elif choice == '4':
        # 运行A*算法
        demo = TrajectoryPlanningDemo()
        demo.run_astar()
        plt.show()

    elif choice == '5':
        # 创建动画
        anim_demo = InteractiveTrajectoryDemo()

        print("\n选择要制作动画的算法:")
        print("1. RRT")
        print("2. PRM")
        print("3. A*")

        algo_choice = input("请输入选项 (1-3): ")

        if algo_choice == '1':
            # 先运行RRT算法
            demo = TrajectoryPlanningDemo()
            demo.run_rrt()
            anim_demo.demo = demo
            anim_demo.create_animation('RRT')
        elif algo_choice == '2':
            # 先运行PRM算法
            demo = TrajectoryPlanningDemo()
            demo.run_prm()
            anim_demo.demo = demo
            anim_demo.create_animation('PRM')
        elif algo_choice == '3':
            # 先运行A*算法
            demo = TrajectoryPlanningDemo()
            demo.run_astar()
            anim_demo.demo = demo
            anim_demo.create_animation('A*')
        else:
            print("无效选项")

    elif choice == '6':
        print("退出程序")
        return

    else:
        print("无效选项")


if __name__ == "__main__":
    main()