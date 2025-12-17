"""
路径规划模块
实现动态路径规划、A*算法、RRT算法等
"""
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import math
import random
from queue import PriorityQueue
from typing import List, Tuple, Optional

class Node:
    """路径节点"""
    
    def __init__(self, x: float, y: float, z: float, parent=None, cost: float = 0.0, heuristic: float = 0.0):
        self.x = x
        self.y = y
        self.z = z
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return (abs(self.x - other.x) < 1e-6 and 
                abs(self.y - other.y) < 1e-6 and 
                abs(self.z - other.z) < 1e-6)
    
    def __hash__(self):
        return hash((self.x, self.y, self.z))
    
    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)
    
    def distance_to(self, other: 'Node') -> float:
        """计算到另一个节点的欧几里得距离"""
        return np.sqrt((self.x - other.x)**2 + 
                      (self.y - other.y)**2 + 
                      (self.z - other.z)**2)
    
    def to_dict(self):
        """转换为字典"""
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'cost': self.cost
        }
    
    def __repr__(self):
        return f"Node({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"

class Obstacle:
    """障碍物"""
    
    def __init__(self, position: List[float], radius: float, height: float):
        self.position = position
        self.radius = radius
        self.height = height
    
    def contains(self, node: Node) -> bool:
        """检查节点是否在障碍物内"""
        dx = node.x - self.position[0]
        dy = node.y - self.position[1]
        dz = node.z - self.position[2]
        horizontal_dist = np.sqrt(dx**2 + dy**2)
        
        if horizontal_dist > self.radius:
            return False
        
        # 检查高度
        if dz < 0 or dz > self.height:
            return False
        
        return True
    
    def distance_to(self, node: Node) -> float:
        """计算到障碍物的最短距离"""
        dx = node.x - self.position[0]
        dy = node.y - self.position[1]
        horizontal_dist = np.sqrt(dx**2 + dy**2) - self.radius
        vertical_dist = min(abs(node.z - self.position[2]), 
                          abs(node.z - (self.position[2] + self.height)))
        
        if horizontal_dist > 0:
            return horizontal_dist if vertical_dist <= 0 else np.sqrt(horizontal_dist**2 + vertical_dist**2)
        else:
            return abs(vertical_dist)

class DynamicPathPlanner:
    """动态路径规划器"""
    
    def __init__(self, grid_size: float = 0.5, safety_margin: float = 1.0):
        """
        初始化路径规划器
        
        Args:
            grid_size: 网格大小（米）
            safety_margin: 安全边界（米）
        """
        self.grid_size = grid_size
        self.safety_margin = safety_margin
        self.obstacles: List[Obstacle] = []
        self.bounds_min = [0, 0, 0]
        self.bounds_max = [100, 100, 50]
        
    def set_environment_bounds(self, min_bound: List[float], max_bound: List[float]):
        """设置环境边界"""
        self.bounds_min = min_bound
        self.bounds_max = max_bound
        
    def add_obstacle(self, obstacle: Obstacle):
        """添加障碍物"""
        self.obstacles.append(obstacle)
        
    def remove_obstacle(self, index: int):
        """移除障碍物"""
        if 0 <= index < len(self.obstacles):
            self.obstacles.pop(index)
            
    def is_collision_free(self, node: Node) -> bool:
        """检查节点是否无碰撞"""
        # 检查边界
        if (node.x < self.bounds_min[0] or node.x > self.bounds_max[0] or
            node.y < self.bounds_min[1] or node.y > self.bounds_max[1] or
            node.z < self.bounds_min[2] or node.z > self.bounds_max[2]):
            return False
        
        # 检查障碍物
        for obstacle in self.obstacles:
            if obstacle.contains(node):
                return False
            
            # 检查安全边界
            if obstacle.distance_to(node) < self.safety_margin:
                return False
                
        return True
    
    def get_neighbors(self, node: Node) -> List[Node]:
        """获取邻居节点"""
        neighbors = []
        
        # 26方向连接（包括对角线）
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                        
                    new_x = node.x + dx * self.grid_size
                    new_y = node.y + dy * self.grid_size
                    new_z = node.z + dz * self.grid_size
                    
                    neighbor = Node(new_x, new_y, new_z, parent=node)
                    
                    # 对角线距离更远
                    move_cost = np.sqrt(dx**2 + dy**2 + dz**2) * self.grid_size
                    neighbor.cost = node.cost + move_cost
                    
                    if self.is_collision_free(neighbor):
                        neighbors.append(neighbor)
                        
        return neighbors
    
    def heuristic(self, node: Node, goal: Node) -> float:
        """启发式函数（欧几里得距离）"""
        return node.distance_to(goal)
    
    def a_star_search(self, start: Node, goal: Node, max_iterations: int = 10000) -> Optional[List[Node]]:
        """
        A*搜索算法
        
        Args:
            start: 起点
            goal: 终点
            max_iterations: 最大迭代次数
            
        Returns:
            路径节点列表（从起点到终点）
        """
        # 初始化开放集和关闭集
        open_set = PriorityQueue()
        closed_set = set()
        
        # 设置起点的启发值
        start.heuristic = self.heuristic(start, goal)
        open_set.put((start.cost + start.heuristic, start))
        
        iterations = 0
        
        while not open_set.empty() and iterations < max_iterations:
            iterations += 1
            
            # 获取当前节点
            _, current = open_set.get()
            
            # 检查是否到达目标
            if current.distance_to(goal) < self.grid_size:
                # 重建路径
                path = []
                while current is not None:
                    path.append(current)
                    current = current.parent
                return list(reversed(path))
            
            # 添加到关闭集
            closed_set.add((current.x, current.y, current.z))
            
            # 获取邻居
            neighbors = self.get_neighbors(current)
            
            for neighbor in neighbors:
                neighbor_key = (neighbor.x, neighbor.y, neighbor.z)
                
                if neighbor_key in closed_set:
                    continue
                
                # 计算启发值
                neighbor.heuristic = self.heuristic(neighbor, goal)
                
                # 添加到开放集
                open_set.put((neighbor.cost + neighbor.heuristic, neighbor))
        
        return None
    
    def rrt_plan(self, start: Node, goal: Node, max_iterations: int = 5000, 
                goal_bias: float = 0.1, step_size: float = 2.0) -> Optional[List[Node]]:
        """
        RRT（快速探索随机树）算法
        
        Args:
            start: 起点
            goal: 终点
            max_iterations: 最大迭代次数
            goal_bias: 目标偏置概率
            step_size: 步长
            
        Returns:
            路径节点列表
        """
        # 初始化树
        tree = [start]
        
        for i in range(max_iterations):
            # 随机采样（有一定概率采样到目标点）
            if random.random() < goal_bias:
                sample = goal
            else:
                sample = Node(
                    random.uniform(self.bounds_min[0], self.bounds_max[0]),
                    random.uniform(self.bounds_min[1], self.bounds_max[1]),
                    random.uniform(self.bounds_min[2], self.bounds_max[2])
                )
            
            # 找到树上最近的节点
            nearest = min(tree, key=lambda n: n.distance_to(sample))
            
            # 向采样点移动
            direction = np.array([sample.x - nearest.x, 
                                 sample.y - nearest.y, 
                                 sample.z - nearest.z])
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm > 0:
                direction = direction / direction_norm
                
                new_node = Node(
                    nearest.x + direction[0] * step_size,
                    nearest.y + direction[1] * step_size,
                    nearest.z + direction[2] * step_size,
                    parent=nearest
                )
                
                # 检查是否无碰撞
                if self.is_collision_free(new_node):
                    # 检查是否可以直接连接到目标
                    if new_node.distance_to(goal) < step_size and self.is_collision_free(goal):
                        goal.parent = new_node
                        tree.append(new_node)
                        tree.append(goal)
                        
                        # 重建路径
                        path = []
                        current = goal
                        while current is not None:
                            path.append(current)
                            current = current.parent
                        return list(reversed(path))
                    
                    tree.append(new_node)
        
        return None
    
    def hybrid_plan(self, start: Node, goal: Node, primary_method: str = 'astar') -> Optional[List[Node]]:
        """
        混合规划算法
        
        Args:
            start: 起点
            goal: 终点
            primary_method: 主要方法 ('astar' 或 'rrt')
            
        Returns:
            路径节点列表
        """
        # 首先尝试A*算法
        if primary_method == 'astar':
            path = self.a_star_search(start, goal)
            if path:
                return path
            
            # 如果A*失败，尝试RRT
            print("A*搜索失败，尝试RRT算法...")
            return self.rrt_plan(start, goal)
        else:
            # 首先尝试RRT算法
            path = self.rrt_plan(start, goal)
            if path:
                return path
            
            # 如果RRT失败，尝试A*
            print("RRT搜索失败，尝试A*算法...")
            return self.a_star_search(start, goal)
    
    def smooth_path(self, path: List[Node], iterations: int = 50) -> List[Node]:
        """
        路径平滑处理
        
        Args:
            path: 原始路径
            iterations: 平滑迭代次数
            
        Returns:
            平滑后的路径
        """
        if len(path) < 3:
            return path
        
        smoothed_path = path.copy()
        
        for _ in range(iterations):
            for i in range(1, len(smoothed_path) - 1):
                # 尝试将当前点向相邻两点中点移动
                prev_node = smoothed_path[i-1]
                next_node = smoothed_path[i+1]
                current_node = smoothed_path[i]
                
                # 计算中点
                mid_x = (prev_node.x + next_node.x) / 2
                mid_y = (prev_node.y + next_node.y) / 2
                mid_z = (prev_node.z + next_node.z) / 2
                
                # 创建新节点
                new_node = Node(mid_x, mid_y, mid_z, parent=current_node.parent)
                
                # 检查新节点是否无碰撞
                if self.is_collision_free(new_node):
                    # 检查新路径是否更短且无碰撞
                    if (new_node.distance_to(prev_node) + new_node.distance_to(next_node) < 
                        current_node.distance_to(prev_node) + current_node.distance_to(next_node)):
                        smoothed_path[i] = new_node
        
        # 更新父节点关系
        for i in range(1, len(smoothed_path)):
            smoothed_path[i].parent = smoothed_path[i-1]
            
        return smoothed_path
    
    def visualize_path(self, path: List[Node], title: str = "路径规划结果"):
        """可视化路径"""
        fig = plt.figure(figsize=(12, 10))
        
        # 创建3D坐标系
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制障碍物
        for obstacle in self.obstacles:
            # 绘制圆柱体
            u = np.linspace(0, 2 * np.pi, 50)
            v = np.linspace(0, obstacle.height, 10)
            u_grid, v_grid = np.meshgrid(u, v)
            
            X = obstacle.radius * np.cos(u_grid) + obstacle.position[0]
            Y = obstacle.radius * np.sin(u_grid) + obstacle.position[1]
            Z = v_grid + obstacle.position[2]
            
            ax.plot_surface(X, Y, Z, alpha=0.3, color='red')
            
            # 绘制安全边界
            X_safe = (obstacle.radius + self.safety_margin) * np.cos(u) + obstacle.position[0]
            Y_safe = (obstacle.radius + self.safety_margin) * np.sin(u) + obstacle.position[1]
            
            ax.plot(X_safe, Y_safe, obstacle.position[2], 'r--', alpha=0.5, linewidth=0.5)
            ax.plot(X_safe, Y_safe, obstacle.position[2] + obstacle.height, 'r--', alpha=0.5, linewidth=0.5)
        
        # 绘制路径
        if path:
            xs = [node.x for node in path]
            ys = [node.y for node in path]
            zs = [node.z for node in path]
            
            ax.plot(xs, ys, zs, 'b-', linewidth=2, label='路径')
            ax.scatter(xs, ys, zs, c='blue', s=20)
            
            # 标记起点和终点
            ax.scatter([xs[0]], [ys[0]], [zs[0]], c='green', s=100, marker='o', label='起点')
            ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], c='red', s=100, marker='x', label='终点')
        
        # 设置坐标轴标签
        ax.set_xlabel('X (米)')
        ax.set_ylabel('Y (米)')
        ax.set_zlabel('Z (米)')
        
        # 设置标题
        ax.set_title(title)
        
        # 添加图例
        ax.legend()
        
        # 设置坐标轴范围
        ax.set_xlim(self.bounds_min[0], self.bounds_max[0])
        ax.set_ylim(self.bounds_min[1], self.bounds_max[1])
        ax.set_zlim(self.bounds_min[2], self.bounds_max[2])
        
        # 显示图形
        plt.tight_layout()
        plt.show()
    
    def export_path_to_json(self, path: List[Node], filepath: str):
        """将路径导出为JSON文件"""
        path_data = {
            'metadata': {
                'grid_size': self.grid_size,
                'safety_margin': self.safety_margin,
                'obstacle_count': len(self.obstacles),
                'generation_time': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'bounds': {
                'min': self.bounds_min,
                'max': self.bounds_max
            },
            'obstacles': [
                {
                    'position': obs.position,
                    'radius': obs.radius,
                    'height': obs.height
                } for obs in self.obstacles
            ],
            'path': [node.to_dict() for node in path]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(path_data, f, indent=2, ensure_ascii=False)
        
        print(f"路径已导出到: {filepath}")

class PathFollower:
    """路径跟随器"""
    
    def __init__(self, lookahead_distance: float = 1.0, max_velocity: float = 2.0):
        """
        初始化路径跟随器
        
        Args:
            lookahead_distance: 前瞻距离（米）
            max_velocity: 最大速度（米/秒）
        """
        self.lookahead_distance = lookahead_distance
        self.max_velocity = max_velocity
        self.path = []
        self.current_waypoint_index = 0
        
    def set_path(self, path: List[Node]):
        """设置路径"""
        self.path = path
        self.current_waypoint_index = 0
        
    def get_next_target(self, current_position: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        获取下一个目标点
        
        Args:
            current_position: 当前位置 [x, y, z]
            
        Returns:
            (target_position, completed) - 目标位置和是否完成
        """
        if not self.path or self.current_waypoint_index >= len(self.path):
            return current_position, True
        
        # 找到最近的路点
        min_dist = float('inf')
        nearest_index = self.current_waypoint_index
        
        for i in range(self.current_waypoint_index, len(self.path)):
            node = self.path[i]
            dist = np.linalg.norm(current_position - np.array([node.x, node.y, node.z]))
            
            if dist < min_dist:
                min_dist = dist
                nearest_index = i
        
        self.current_waypoint_index = nearest_index
        
        # 寻找前瞻点
        lookahead_index = nearest_index
        accumulated_distance = 0.0
        
        for i in range(nearest_index, len(self.path) - 1):
            node1 = self.path[i]
            node2 = self.path[i + 1]
            segment_length = node1.distance_to(node2)
            
            if accumulated_distance + segment_length >= self.lookahead_distance:
                # 在这个线段上找到前瞻点
                ratio = (self.lookahead_distance - accumulated_distance) / segment_length
                target = Node(
                    node1.x + ratio * (node2.x - node1.x),
                    node1.y + ratio * (node2.y - node1.y),
                    node1.z + ratio * (node2.z - node1.z)
                )
                return np.array([target.x, target.y, target.z]), False
            
            accumulated_distance += segment_length
            lookahead_index = i + 1
        
        # 如果到达终点
        last_node = self.path[-1]
        return np.array([last_node.x, last_node.y, last_node.z]), True
    
    def compute_control_command(self, current_position: np.ndarray, 
                               current_velocity: np.ndarray,
                               target_position: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算控制命令
        
        Args:
            current_position: 当前位置
            current_velocity: 当前速度
            target_position: 目标位置
            
        Returns:
            (desired_velocity, desired_acceleration)
        """
        # 计算位置误差
        position_error = target_position - current_position
        distance = np.linalg.norm(position_error)
        
        # 计算期望速度（PD控制器）
        kp = 2.0  # 比例增益
        kd = 0.5  # 微分增益
        
        desired_direction = position_error / distance if distance > 0 else np.zeros(3)
        
        # 速度限制
        desired_speed = min(self.max_velocity, kp * distance)
        desired_velocity = desired_direction * desired_speed
        
        # 计算期望加速度
        velocity_error = desired_velocity - current_velocity
        desired_acceleration = kp * position_error + kd * velocity_error
        
        # 加速度限制
        max_acceleration = 2.0
        if np.linalg.norm(desired_acceleration) > max_acceleration:
            desired_acceleration = desired_acceleration / np.linalg.norm(desired_acceleration) * max_acceleration
        
        return desired_velocity, desired_acceleration
    
    def get_progress(self) -> float:
        """获取进度百分比"""
        if not self.path or len(self.path) < 2:
            return 0.0
        
        return min(100.0, (self.current_waypoint_index / len(self.path)) * 100)