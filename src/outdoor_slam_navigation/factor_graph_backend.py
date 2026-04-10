# factor_graph_backend.py
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import time
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
# 在 factor_graph_backend.py 文件开头，import语句后面添加

import matplotlib.pyplot as plt

# 设置中文字体
try:
    # Windows系统字体路径
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    # Mac系统字体路径
    # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'STHeiti', 'Heiti SC']
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
except:
    print("警告：无法加载中文字体，将使用默认字体")
    pass

# 设置matplotlib默认参数
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


@dataclass
class Factor:
    """因子基类"""
    factor_type: str
    nodes: List[int]  # 关联的节点ID
    measurement: np.ndarray
    information: np.ndarray  # 信息矩阵


@dataclass
class Node:
    """位姿节点"""
    node_id: int
    timestamp: float
    pose: np.ndarray  # 4x4变换矩阵
    fixed: bool = False


class FactorGraphVisualizer:
    """因子图可视化类"""

    def __init__(self, backend: 'FactorGraphBackend'):
        self.backend = backend
        self.fig = None
        self.ax = None

    def plot_trajectory_2d(self, show_factor_graph: bool = True,
                           show_submaps: bool = True,
                           show_loop_closures: bool = True,
                           save_path: Optional[str] = None):
        """绘制2D轨迹图"""
        if len(self.backend.nodes) == 0:
            print("没有节点可显示")
            return

        # 创建图形
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 7))

        # 绘制轨迹
        self._plot_trajectory_2d_on_axis(self.ax1, "优化前轨迹", show_factor_graph, show_submaps, show_loop_closures)

        # 如果有优化结果，绘制优化后的轨迹
        if hasattr(self.backend, 'optimized_poses') and self.backend.optimized_poses:
            self._plot_optimized_trajectory_2d(self.ax2)
        else:
            # 否则绘制节点和因子图
            self._plot_trajectory_2d_on_axis(self.ax2, "节点和因子图", True, False, False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"轨迹图已保存到 {save_path}")

        plt.show()

    def _plot_trajectory_2d_on_axis(self, ax, title: str,
                                    show_factor_graph: bool,
                                    show_submaps: bool,
                                    show_loop_closures: bool):
        """在指定轴上绘制轨迹"""
        # 提取轨迹点
        x_coords = []
        y_coords = []
        timestamps = []

        for node_id in sorted(self.backend.nodes.keys()):
            node = self.backend.nodes[node_id]
            pos = node.pose[:3, 3]
            x_coords.append(pos[0])
            y_coords.append(pos[1])
            timestamps.append(node.timestamp)

        # 绘制轨迹线
        ax.plot(x_coords, y_coords, 'b-', alpha=0.5, linewidth=1, label='轨迹')

        # 绘制节点点
        ax.scatter(x_coords, y_coords, c=timestamps, cmap='viridis',
                   s=20, alpha=0.7, label='节点')

        # 绘制固定节点（如果有）
        fixed_x = []
        fixed_y = []
        for node_id, node in self.backend.nodes.items():
            if node.fixed:
                pos = node.pose[:3, 3]
                fixed_x.append(pos[0])
                fixed_y.append(pos[1])

        if fixed_x:
            ax.scatter(fixed_x, fixed_y, c='red', s=100,
                       marker='*', label='固定节点', zorder=5)

        # 绘制因子图连接
        if show_factor_graph:
            self._plot_factor_connections_2d(ax)

        # 绘制子图
        if show_submaps:
            self._plot_submaps_2d(ax)

        # 绘制闭环
        if show_loop_closures:
            self._plot_loop_closures_2d(ax)

        ax.set_xlabel('X (米)')
        ax.set_ylabel('Y (米)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axis('equal')

    def _plot_factor_connections_2d(self, ax):
        """绘制因子连接"""
        # 定义颜色映射
        factor_colors = {
            'odometry': 'green',
            'gnss': 'blue',
            'loop_closure': 'red'
        }

        # 绘制因子连接
        for factor in self.backend.factors:
            if len(factor.nodes) >= 2:
                color = factor_colors.get(factor.factor_type, 'gray')
                linewidth = 1.0
                alpha = 0.3

                # 不同类型因子使用不同样式
                if factor.factor_type == 'loop_closure':
                    linewidth = 1.5
                    alpha = 0.8

                # 获取节点位置
                positions = []
                for node_id in factor.nodes:
                    if node_id in self.backend.nodes:
                        pos = self.backend.nodes[node_id].pose[:3, 3]
                        positions.append(pos[:2])  # 只取XY

                if len(positions) >= 2:
                    # 绘制连接线
                    for i in range(len(positions) - 1):
                        ax.plot([positions[i][0], positions[i + 1][0]],
                                [positions[i][1], positions[i + 1][1]],
                                color=color, linewidth=linewidth,
                                alpha=alpha, linestyle='--')

    def _plot_submaps_2d(self, ax):
        """绘制子图区域"""
        for i, submap in enumerate(self.backend.submaps):
            if submap["nodes"]:
                # 提取子图节点位置
                positions = []
                for node_id in submap["nodes"]:
                    if node_id in self.backend.nodes:
                        pos = self.backend.nodes[node_id].pose[:3, 3]
                        positions.append(pos[:2])

                if positions:
                    positions = np.array(positions)
                    # 计算边界框
                    x_min, y_min = positions.min(axis=0)
                    x_max, y_max = positions.max(axis=0)

                    # 绘制矩形框
                    rect = plt.Rectangle((x_min, y_min),
                                         x_max - x_min,
                                         y_max - y_min,
                                         fill=False,
                                         edgecolor='orange',
                                         linewidth=1,
                                         alpha=0.5,
                                         linestyle=':')
                    ax.add_patch(rect)

                    # 添加子图编号
                    ax.text(x_min, y_min, f'Submap {i}',
                            fontsize=8, color='orange',
                            bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor="white",
                                      alpha=0.7))

    def _plot_loop_closures_2d(self, ax):
        """绘制闭环连接"""
        # 从因子中提取闭环
        for factor in self.backend.factors:
            if factor.factor_type == 'loop_closure' and len(factor.nodes) >= 2:
                node_i, node_j = factor.nodes[:2]
                if node_i in self.backend.nodes and node_j in self.backend.nodes:
                    pos_i = self.backend.nodes[node_i].pose[:3, 3][:2]
                    pos_j = self.backend.nodes[node_j].pose[:3, 3][:2]

                    # 绘制闭环连接线
                    ax.plot([pos_i[0], pos_j[0]],
                            [pos_i[1], pos_j[1]],
                            color='red', linewidth=2,
                            alpha=0.8, linestyle='-',
                            label='闭环' if not ax.get_legend() else '')

    def _plot_optimized_trajectory_2d(self, ax):
        """绘制优化后的轨迹"""
        # 提取优化前后的轨迹
        orig_x, orig_y = [], []
        opt_x, opt_y = [], []

        for node_id in sorted(self.backend.nodes.keys()):
            # 原始位置
            orig_pos = self.backend.nodes[node_id].pose[:3, 3]
            orig_x.append(orig_pos[0])
            orig_y.append(orig_pos[1])

            # 优化后位置
            if node_id in self.backend.optimized_poses:
                opt_pos = self.backend.optimized_poses[node_id][:3, 3]
                opt_x.append(opt_pos[0])
                opt_y.append(opt_pos[1])

        # 绘制轨迹
        ax.plot(orig_x, orig_y, 'b-', alpha=0.3, linewidth=1, label='优化前')
        ax.plot(opt_x, opt_y, 'r-', alpha=0.8, linewidth=2, label='优化后')

        # 绘制节点
        ax.scatter(orig_x, orig_y, c='blue', s=20, alpha=0.3, label='原节点')
        ax.scatter(opt_x, opt_y, c='red', s=30, alpha=0.6, label='优化节点')

        # 绘制修正箭头
        for i in range(len(orig_x)):
            if i < len(opt_x):
                ax.arrow(orig_x[i], orig_y[i],
                         opt_x[i] - orig_x[i],
                         opt_y[i] - orig_y[i],
                         head_width=0.1, head_length=0.2,
                         fc='green', ec='green', alpha=0.3)

        ax.set_xlabel('X (米)')
        ax.set_ylabel('Y (米)')
        ax.set_title('优化前后轨迹对比')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axis('equal')

    def plot_3d_trajectory(self, show_poses: bool = False,
                           save_path: Optional[str] = None):
        """绘制3D轨迹图"""
        if len(self.backend.nodes) == 0:
            print("没有节点可显示")
            return

        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # 提取轨迹点
        x_coords = []
        y_coords = []
        z_coords = []
        timestamps = []

        for node_id in sorted(self.backend.nodes.keys()):
            node = self.backend.nodes[node_id]
            pos = node.pose[:3, 3]
            x_coords.append(pos[0])
            y_coords.append(pos[1])
            z_coords.append(pos[2])
            timestamps.append(node.timestamp)

        # 绘制轨迹线
        self.ax.plot(x_coords, y_coords, z_coords, 'b-', alpha=0.5, linewidth=1)

        # 绘制节点点
        scatter = self.ax.scatter(x_coords, y_coords, z_coords,
                                  c=timestamps, cmap='viridis',
                                  s=20, alpha=0.7)

        # 绘制位姿方向（可选）
        if show_poses:
            self._plot_3d_poses()

        # 绘制因子连接
        self._plot_3d_factor_connections()

        # 添加颜色条
        cbar = self.fig.colorbar(scatter, ax=self.ax, shrink=0.5, aspect=5)
        cbar.set_label('时间戳')

        self.ax.set_xlabel('X (米)')
        self.ax.set_ylabel('Y (米)')
        self.ax.set_zlabel('Z (米)')
        self.ax.set_title('3D轨迹图')

        # 设置视角
        self.ax.view_init(elev=30, azim=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"3D轨迹图已保存到 {save_path}")

        plt.show()

    def _plot_3d_poses(self):
        """在3D图中绘制位姿方向"""
        for node_id in sorted(self.backend.nodes.keys()):
            if node_id % 5 == 0:  # 每5个节点绘制一个方向
                node = self.backend.nodes[node_id]
                pose = node.pose

                # 位置
                pos = pose[:3, 3]

                # X方向（红色）
                x_dir = pose[:3, 0]
                self.ax.quiver(pos[0], pos[1], pos[2],
                               x_dir[0] * 0.5, x_dir[1] * 0.5, x_dir[2] * 0.5,
                               color='r', linewidth=1, alpha=0.6)

                # Y方向（绿色）
                y_dir = pose[:3, 1]
                self.ax.quiver(pos[0], pos[1], pos[2],
                               y_dir[0] * 0.5, y_dir[1] * 0.5, y_dir[2] * 0.5,
                               color='g', linewidth=1, alpha=0.6)

                # Z方向（蓝色）
                z_dir = pose[:3, 2]
                self.ax.quiver(pos[0], pos[1], pos[2],
                               z_dir[0] * 0.5, z_dir[1] * 0.5, z_dir[2] * 0.5,
                               color='b', linewidth=1, alpha=0.6)

    def _plot_3d_factor_connections(self):
        """在3D图中绘制因子连接"""
        factor_colors = {
            'odometry': 'green',
            'gnss': 'blue',
            'loop_closure': 'red'
        }

        for factor in self.backend.factors:
            if len(factor.nodes) >= 2:
                color = factor_colors.get(factor.factor_type, 'gray')
                alpha = 0.3
                linewidth = 1

                if factor.factor_type == 'loop_closure':
                    alpha = 0.8
                    linewidth = 2

                # 获取节点位置
                positions = []
                for node_id in factor.nodes:
                    if node_id in self.backend.nodes:
                        pos = self.backend.nodes[node_id].pose[:3, 3]
                        positions.append(pos)

                if len(positions) >= 2:
                    for i in range(len(positions) - 1):
                        x = [positions[i][0], positions[i + 1][0]]
                        y = [positions[i][1], positions[i + 1][1]]
                        z = [positions[i][2], positions[i + 1][2]]

                        self.ax.plot(x, y, z, color=color,
                                     linewidth=linewidth, alpha=alpha,
                                     linestyle='--')

    def plot_error_evolution(self, save_path: Optional[str] = None):
        """绘制误差演化图"""
        if not hasattr(self.backend, 'optimization_history'):
            print("没有优化历史数据")
            return

        history = self.backend.optimization_history

        self.fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 总误差
        if 'total_errors' in history:
            ax = axes[0, 0]
            ax.plot(history['total_errors'], 'b-', linewidth=2)
            ax.set_xlabel('迭代次数')
            ax.set_ylabel('总误差')
            ax.set_title('总误差收敛曲线')
            ax.grid(True, alpha=0.3)

        # 位置误差
        if 'position_errors' in history:
            ax = axes[0, 1]
            ax.plot(history['position_errors'], 'g-', linewidth=2)
            ax.set_xlabel('迭代次数')
            ax.set_ylabel('位置误差 (米)')
            ax.set_title('位置误差收敛曲线')
            ax.grid(True, alpha=0.3)

        # 旋转误差
        if 'rotation_errors' in history:
            ax = axes[1, 0]
            ax.plot(history['rotation_errors'], 'r-', linewidth=2)
            ax.set_xlabel('迭代次数')
            ax.set_ylabel('旋转误差 (弧度)')
            ax.set_title('旋转误差收敛曲线')
            ax.grid(True, alpha=0.3)

        # 梯度范数
        if 'gradient_norms' in history:
            ax = axes[1, 1]
            ax.plot(history['gradient_norms'], 'm-', linewidth=2)
            ax.set_xlabel('迭代次数')
            ax.set_ylabel('梯度范数')
            ax.set_title('梯度范数变化')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"误差演化图已保存到 {save_path}")

        plt.show()

    def animate_trajectory(self, interval: float = 100,
                           save_path: Optional[str] = None):
        """创建轨迹动画"""
        if len(self.backend.nodes) < 2:
            print("节点数量不足，无法创建动画")
            return

        self.fig, self.ax = plt.subplots(figsize=(10, 8))

        # 提取所有位置
        x_coords = []
        y_coords = []
        node_ids = sorted(self.backend.nodes.keys())

        for node_id in node_ids:
            pos = self.backend.nodes[node_id].pose[:3, 3]
            x_coords.append(pos[0])
            y_coords.append(pos[1])

        # 初始化动画元素
        line, = self.ax.plot([], [], 'b-', linewidth=2, alpha=0.8)
        point, = self.ax.plot([], [], 'ro', markersize=8, alpha=0.8)
        timestamp_text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes)
        node_text = self.ax.text(0.05, 0.90, '', transform=self.ax.transAxes)

        # 设置图形范围
        margin = 5.0
        self.ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
        self.ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
        self.ax.set_xlabel('X (米)')
        self.ax.set_ylabel('Y (米)')
        self.ax.set_title('轨迹生成动画')
        self.ax.grid(True, alpha=0.3)
        self.ax.axis('equal')

        def init():
            """初始化动画"""
            line.set_data([], [])
            point.set_data([], [])
            timestamp_text.set_text('')
            node_text.set_text('')
            return line, point, timestamp_text, node_text

        def animate(i):
            """动画更新函数"""
            # 更新轨迹线
            line.set_data(x_coords[:i + 1], y_coords[:i + 1])

            # 更新当前点
            if i < len(x_coords):
                point.set_data([x_coords[i]], [y_coords[i]])

            # 更新文本
            if i < len(node_ids):
                node = self.backend.nodes[node_ids[i]]
                timestamp_text.set_text(f'时间戳: {node.timestamp:.2f}s')
                node_text.set_text(f'节点ID: {node.node_id}')

            return line, point, timestamp_text, node_text

        # 创建动画
        anim = animation.FuncAnimation(self.fig, animate, init_func=init,
                                       frames=len(node_ids),
                                       interval=interval, blit=True)

        plt.tight_layout()

        # 保存动画
        if save_path:
            anim.save(save_path, writer='ffmpeg', fps=30)
            print(f"轨迹动画已保存到 {save_path}")

        plt.show()
        return anim


class FactorGraphBackend:
    """因子图优化后端（带可视化功能）"""

    def __init__(self, config=None):
        self.config = config or self._default_config()

        # 图结构
        self.nodes = {}  # node_id -> Node
        self.factors = []  # List[Factor]

        # 子图管理
        self.submaps = []
        self.current_submap = None

        # 优化参数
        self.optimization_interval = 10  # 每10帧优化一次
        self.frame_count = 0

        # GNSS融合
        self.gnss_reference = None  # 局部坐标原点

        # 优化历史
        self.optimization_history = {
            'total_errors': [],
            'position_errors': [],
            'rotation_errors': [],
            'gradient_norms': []
        }

        # 可视化器
        self.visualizer = FactorGraphVisualizer(self)

        # 优化结果缓存
        self.optimized_poses = {}

    def _default_config(self):
        return {
            "optimization": {
                "max_iterations": 50,
                "convergence_threshold": 1e-6,
                "use_sparse": True
            },
            "loop_closure": {
                "search_radius": 10.0,  # 米
                "min_chain_length": 10,
                "score_threshold": 0.5
            },
            "submap": {
                "size": 100,  # 每100帧一个子图
                "overlap": 20  # 子图间重叠帧数
            },
            "visualization": {
                "show_trajectory": True,
                "show_factors": True,
                "show_submaps": True,
                "show_loop_closures": True
            }
        }

    def add_node(self, pose: np.ndarray, timestamp: float,
                 node_type: str = "odometry", fixed: bool = False) -> int:
        """添加位姿节点"""
        node_id = len(self.nodes)
        node = Node(
            node_id=node_id,
            timestamp=timestamp,
            pose=pose.copy(),
            fixed=fixed
        )
        self.nodes[node_id] = node

        # 添加到当前子图
        if self.current_submap is None:
            self.current_submap = {
                "start_id": node_id,
                "end_id": node_id,
                "nodes": [node_id],
                "features": []
            }
        else:
            self.current_submap["nodes"].append(node_id)
            self.current_submap["end_id"] = node_id

            # 检查子图是否完成
            if len(self.current_submap["nodes"]) >= self.config["submap"]["size"]:
                self.submaps.append(self.current_submap)
                # 创建新子图，保持重叠
                overlap_nodes = self.current_submap["nodes"][-self.config["submap"]["overlap"]:]
                self.current_submap = {
                    "start_id": overlap_nodes[0],
                    "end_id": node_id,
                    "nodes": overlap_nodes.copy(),
                    "features": []
                }

        return node_id

    def add_odometry_factor(self, node_i: int, node_j: int,
                            relative_pose: np.ndarray, covariance: np.ndarray):
        """添加里程计因子"""
        info_matrix = np.linalg.inv(covariance)

        factor = Factor(
            factor_type="odometry",
            nodes=[node_i, node_j],
            measurement=relative_pose,
            information=info_matrix
        )
        self.factors.append(factor)

    def add_gnss_factor(self, node_id: int, gnss_position: np.ndarray,
                        covariance: np.ndarray):
        """添加GNSS因子"""
        if self.gnss_reference is None:
            self.gnss_reference = gnss_position.copy()

        # 转换为局部坐标
        local_pos = gnss_position - self.gnss_reference

        # 创建位置测量（只有位置，没有姿态）
        measurement = np.eye(4)
        measurement[:3, 3] = local_pos

        info_matrix = np.linalg.inv(covariance)

        factor = Factor(
            factor_type="gnss",
            nodes=[node_id],
            measurement=measurement,
            information=info_matrix
        )
        self.factors.append(factor)

    def add_loop_closure_factor(self, node_i: int, node_j: int,
                                relative_pose: np.ndarray, covariance: np.ndarray):
        """添加闭环因子"""
        info_matrix = np.linalg.inv(covariance)

        factor = Factor(
            factor_type="loop_closure",
            nodes=[node_i, node_j],
            measurement=relative_pose,
            information=info_matrix
        )
        self.factors.append(factor)

    def detect_loop_closure(self, current_node_id: int, features: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """检测闭环"""
        if len(self.submaps) < 2:
            return []

        loops = []
        current_node = self.nodes[current_node_id]
        current_pos = current_node.pose[:3, 3]

        # 只在历史子图中搜索
        for submap in self.submaps[:-1]:  # 排除当前子图
            # 距离检查
            submap_center = self._get_submap_center(submap)
            distance = np.linalg.norm(current_pos - submap_center)

            if distance < self.config["loop_closure"]["search_radius"]:
                # 特征匹配
                for node_id in submap["nodes"]:
                    if node_id == current_node_id:
                        continue

                    # 时间间隔检查（避免匹配太近的帧）
                    time_diff = abs(current_node.timestamp - self.nodes[node_id].timestamp)
                    if time_diff < 5.0:  # 5秒内不闭环
                        continue

                    # 特征匹配（简化版本）
                    match_score = self._feature_matching_score(features, submap["features"])

                    if match_score > self.config["loop_closure"]["score_threshold"]:
                        # 计算相对位姿
                        relative_pose = self._compute_relative_pose(
                            current_node.pose,
                            self.nodes[node_id].pose
                        )
                        loops.append((node_id, relative_pose))

        return loops

    def _feature_matching_score(self, features1, features2):
        """计算特征匹配分数（简化版）"""
        if len(features1) == 0 or len(features2) == 0:
            return 0.0

        # 在实际系统中，这里应该使用更复杂的特征描述子匹配
        # 这里返回一个随机分数作为示例
        return np.random.uniform(0, 1)

    def _compute_relative_pose(self, pose_i, pose_j):
        """计算相对位姿"""
        return np.linalg.inv(pose_j) @ pose_i

    def _get_submap_center(self, submap):
        """获取子图中心位置"""
        positions = []
        for node_id in submap["nodes"]:
            if node_id in self.nodes:
                positions.append(self.nodes[node_id].pose[:3, 3])

        if positions:
            return np.mean(positions, axis=0)
        else:
            return np.zeros(3)

    def plot_statistics(self):
            """绘制统计信息"""
            if len(self.nodes) == 0:
                print("没有数据可统计")
                return

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # 1. 节点数量统计
            ax = axes[0, 0]
            factor_types = {}
            for factor in self.factors:
                factor_types[factor.factor_type] = factor_types.get(factor.factor_type, 0) + 1

            if factor_types:
                colors = plt.cm.Set3(np.linspace(0, 1, len(factor_types)))
                ax.pie(factor_types.values(), labels=factor_types.keys(),
                       autopct='%1.1f%%', colors=colors, startangle=90)
                ax.set_title('因子类型分布')
            else:
                ax.text(0.5, 0.5, '无因子数据', ha='center', va='center')
                ax.set_title('因子类型分布')

            # 2. 轨迹长度统计
            ax = axes[0, 1]
            distances = []
            prev_pos = None
            for node_id in sorted(self.nodes.keys()):
                pos = self.nodes[node_id].pose[:3, 3]
                if prev_pos is not None:
                    distances.append(np.linalg.norm(pos - prev_pos))
                prev_pos = pos

            if distances:
                ax.hist(distances, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_xlabel('帧间距离 (米)')
                ax.set_ylabel('频次')
                ax.set_title('帧间距离分布')
            else:
                ax.text(0.5, 0.5, '无距离数据', ha='center', va='center')
                ax.set_title('帧间距离分布')

            # 3. 时间间隔统计
            ax = axes[0, 2]
            time_intervals = []
            timestamps = sorted([node.timestamp for node in self.nodes.values()])

            if len(timestamps) > 1:
                for i in range(1, len(timestamps)):
                    time_intervals.append(timestamps[i] - timestamps[i - 1])

                ax.plot(time_intervals, 'g-', alpha=0.7)
                ax.fill_between(range(len(time_intervals)), 0, time_intervals,
                                alpha=0.3, color='green')
                ax.set_xlabel('帧序号')
                ax.set_ylabel('时间间隔 (秒)')
                ax.set_title('时间间隔变化')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, '无时间间隔数据', ha='center', va='center')
                ax.set_title('时间间隔变化')

            # 4. 节点分布统计
            ax = axes[1, 0]
            if len(self.nodes) > 0:
                node_ids = list(self.nodes.keys())
                node_types = ['固定' if node.fixed else '非固定' for node in self.nodes.values()]

                fixed_count = sum(1 for node in self.nodes.values() if node.fixed)
                non_fixed_count = len(self.nodes) - fixed_count

                ax.bar(['固定节点', '非固定节点'], [fixed_count, non_fixed_count],
                       color=['red', 'blue'], alpha=0.7)
                ax.set_ylabel('数量')
                ax.set_title('节点类型分布')
                ax.grid(True, alpha=0.3, axis='y')
            else:
                ax.text(0.5, 0.5, '无节点数据', ha='center', va='center')
                ax.set_title('节点类型分布')

            # 5. 子图统计
            ax = axes[1, 1]
            if self.submaps:
                submap_sizes = [len(submap['nodes']) for submap in self.submaps]
                submap_ids = list(range(len(self.submaps)))

                ax.bar(submap_ids, submap_sizes, color='orange', alpha=0.7)
                ax.set_xlabel('子图编号')
                ax.set_ylabel('节点数量')
                ax.set_title('子图大小分布')
                ax.grid(True, alpha=0.3, axis='y')
            else:
                ax.text(0.5, 0.5, '无子图数据', ha='center', va='center')
                ax.set_title('子图大小分布')

            # 6. 优化历史统计（如果有）
            ax = axes[1, 2]
            if hasattr(self, 'optimization_history') and self.optimization_history.get('total_errors'):
                iterations = list(range(len(self.optimization_history['total_errors'])))
                ax.plot(iterations, self.optimization_history['total_errors'],
                        'b-', linewidth=2, label='总误差')

                if self.optimization_history.get('position_errors'):
                    ax.plot(iterations, self.optimization_history['position_errors'],
                            'g-', linewidth=2, label='位置误差', alpha=0.7)

                if self.optimization_history.get('rotation_errors'):
                    ax.plot(iterations, self.optimization_history['rotation_errors'],
                            'r-', linewidth=2, label='旋转误差', alpha=0.7)

                ax.set_xlabel('迭代次数')
                ax.set_ylabel('误差')
                ax.set_title('优化收敛曲线')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
            else:
                ax.text(0.5, 0.5, '无优化历史数据', ha='center', va='center')
                ax.set_title('优化收敛曲线')

            plt.suptitle('因子图统计信息', fontsize=16, fontweight='bold')
            plt.tight_layout()

            # 保存图片
            plt.savefig('factor_graph_statistics.png', dpi=150, bbox_inches='tight')
            print("统计图已保存为 'factor_graph_statistics.png'")

            plt.show()
    def optimize(self):
        """执行图优化"""
        print(f"开始优化，节点数: {len(self.nodes)}，因子数: {len(self.factors)}")

        # 清空历史记录
        self.optimization_history = {
            'total_errors': [],
            'position_errors': [],
            'rotation_errors': [],
            'gradient_norms': []
        }

        # 简化的优化实现
        # 在实际系统中，应该使用g2o、GTSAM或Ceres等优化库

        # 这里实现一个简单的梯度下降优化
        for iteration in range(self.config["optimization"]["max_iterations"]):
            total_error = 0.0
            position_error = 0.0
            rotation_error = 0.0

            # 计算每个节点的梯度
            gradients = {node_id: np.zeros(6) for node_id in self.nodes}

            for factor in self.factors:
                error, jacobians = self._compute_factor_error(factor)
                total_error += np.sum(error ** 2)

                # 计算位置和旋转误差
                if factor.factor_type == 'odometry':
                    position_error += np.linalg.norm(error[3:])
                    rotation_error += np.linalg.norm(error[:3])

                # 更新梯度
                for i, node_id in enumerate(factor.nodes):
                    if node_id in gradients:
                        gradients[node_id] += jacobians[i] @ error

            # 记录历史
            self.optimization_history['total_errors'].append(total_error)
            self.optimization_history['position_errors'].append(position_error)
            self.optimization_history['rotation_errors'].append(rotation_error)

            # 计算平均梯度范数
            grad_norm = 0.0
            for grad in gradients.values():
                grad_norm += np.linalg.norm(grad)
            grad_norm /= len(gradients)
            self.optimization_history['gradient_norms'].append(grad_norm)

            # 更新节点位姿
            for node_id, gradient in gradients.items():
                if not self.nodes[node_id].fixed:
                    # 将梯度转换为位姿更新
                    delta_pose = self._gradient_to_pose(gradient, 0.01)
                    self.nodes[node_id].pose = delta_pose @ self.nodes[node_id].pose

            # 检查收敛
            if total_error < self.config["optimization"]["convergence_threshold"]:
                print(f"优化在第{iteration}次迭代收敛")
                break

        print(f"优化完成，最终误差: {total_error}")

        # 保存优化结果
        self.optimized_poses = {}
        for node_id, node in self.nodes.items():
            self.optimized_poses[node_id] = node.pose.copy()

        # 可视化优化结果
        if self.config["visualization"]["show_trajectory"]:
            self.visualize()

        return self.optimized_poses

    def _compute_factor_error(self, factor):
        """计算因子误差（简化实现）"""
        # 这里应该根据因子类型计算不同的误差
        # 简化实现返回零误差和雅可比
        error = np.zeros(6)
        jacobians = [np.eye(6) for _ in factor.nodes]

        return error, jacobians

    def _gradient_to_pose(self, gradient, step_size):
        """将梯度转换为位姿更新"""
        # 将6维梯度转换为SE(3)变换
        delta_pose = np.eye(4)

        # 旋转部分（前3维）
        rotation_vec = gradient[:3] * step_size
        if np.linalg.norm(rotation_vec) > 0:
            from scipy.spatial.transform import Rotation
            delta_rotation = Rotation.from_rotvec(rotation_vec).as_matrix()
            delta_pose[:3, :3] = delta_rotation

        # 平移部分（后3维）
        delta_pose[:3, 3] = gradient[3:] * step_size

        return delta_pose

    def get_optimized_trajectory(self):
        """获取优化后的轨迹"""
        trajectory = []
        for node_id in sorted(self.nodes.keys()):
            node = self.nodes[node_id]
            trajectory.append({
                "timestamp": node.timestamp,
                "pose": node.pose,
                "position": node.pose[:3, 3]
            })
        return trajectory

    def save_map(self, filename):
        """保存地图"""
        import pickle

        map_data = {
            "nodes": self.nodes,
            "submaps": self.submaps,
            "config": self.config,
            "gnss_reference": self.gnss_reference,
            "optimized_poses": self.optimized_poses,
            "optimization_history": self.optimization_history
        }

        with open(filename, 'wb') as f:
            pickle.dump(map_data, f)

        print(f"地图已保存到 {filename}")

    def load_map(self, filename):
        """加载地图"""
        import pickle

        with open(filename, 'rb') as f:
            map_data = pickle.load(f)

        self.nodes = map_data["nodes"]
        self.submaps = map_data["submaps"]
        self.config = map_data["config"]
        self.gnss_reference = map_data.get("gnss_reference")
        self.optimized_poses = map_data.get("optimized_poses", {})
        self.optimization_history = map_data.get("optimization_history", {})

        print(f"已加载地图，包含 {len(self.nodes)} 个节点")

    # ==================== 可视化方法 ====================

    def visualize(self, show_2d: bool = True, show_3d: bool = False,
                  show_errors: bool = True, show_animation: bool = False,
                  save_prefix: Optional[str] = None):
        """综合可视化功能"""

        if show_2d:
            save_path = f"{save_prefix}_2d.png" if save_prefix else None
            self.visualizer.plot_trajectory_2d(
                show_factor_graph=self.config["visualization"]["show_factors"],
                show_submaps=self.config["visualization"]["show_submaps"],
                show_loop_closures=self.config["visualization"]["show_loop_closures"],
                save_path=save_path
            )

        if show_3d:
            save_path = f"{save_prefix}_3d.png" if save_prefix else None
            self.visualizer.plot_3d_trajectory(save_path=save_path)

        if show_errors and hasattr(self, 'optimization_history'):
            save_path = f"{save_prefix}_errors.png" if save_prefix else None
            self.visualizer.plot_error_evolution(save_path=save_path)

        if show_animation:
            save_path = f"{save_prefix}_animation.mp4" if save_prefix else None
            self.visualizer.animate_trajectory(save_path=save_path)

    def plot_statistics(self):
        """绘制统计信息"""
        if len(self.nodes) == 0:
            print("没有数据可统计")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. 节点数量统计
        ax = axes[0, 0]
        factor_types = {}
        for factor in self.factors:
            factor_types[factor.factor_type] = factor_types.get(factor.factor_type, 0) + 1

        colors = plt.cm.Set3(np.linspace(0, 1, len(factor_types)))
        ax.pie(factor_types.values(), labels=factor_types.keys(),
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('因子类型分布')

        # 2. 轨迹长度统计
        ax = axes[0, 1]
        distances = []
        prev_pos = None
        for node_id in sorted(self.nodes.keys()):
            pos = self.nodes[node_id].pose[:3, 3]
            if prev_pos is not None:
                distances.append(np.linalg.norm(pos - prev_pos))
            prev_pos = pos

        if distances:
            ax.hist(distances, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('帧间距离 (米)')
            ax.set_ylabel('频次')
            ax.set_title('帧间距离分布')


# 在 factor_graph_backend.py 文件末尾添加以下代码

def main():
    """示例演示函数"""
    print("因子图后端可视化演示")

    # 创建因子图后端
    backend = FactorGraphBackend()

    # 添加一些示例节点和因子
    import random

    # 生成一个简单的圆形轨迹
    num_nodes = 50
    radius = 10.0
    center_x, center_y = 0, 0

    for i in range(num_nodes):
        # 生成圆形轨迹上的位姿
        angle = 2 * np.pi * i / num_nodes

        # 位置
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        z = 0.0

        # 姿态（朝向切线方向）
        from scipy.spatial.transform import Rotation

        # 切线方向
        dx = -np.sin(angle)
        dy = np.cos(angle)

        # 创建旋转矩阵
        rotation = Rotation.from_euler('z', angle).as_matrix()

        # 创建位姿矩阵
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = [x, y, z]

        # 添加节点
        node_id = backend.add_node(pose, timestamp=i)

        # 添加里程计因子
        if i > 0:
            # 计算相对位姿
            prev_pose = backend.nodes[i - 1].pose
            relative_pose = np.linalg.inv(prev_pose) @ pose

            # 添加里程计因子
            covariance = np.eye(6) * 0.1  # 协方差矩阵
            backend.add_odometry_factor(i - 1, i, relative_pose, covariance)

    # 添加一些GNSS因子
    for i in range(0, num_nodes, 5):
        gnss_pos = backend.nodes[i].pose[:3, 3] + np.random.randn(3) * 0.5
        covariance = np.eye(3) * 1.0
        backend.add_gnss_factor(i, gnss_pos, covariance)

    # 添加一个闭环因子（模拟回环检测）
    if num_nodes > 20:
        # 连接起点和终点附近的一个点
        loop_i = 0
        loop_j = num_nodes - 10

        pose_i = backend.nodes[loop_i].pose
        pose_j = backend.nodes[loop_j].pose
        relative_pose = np.linalg.inv(pose_j) @ pose_i

        covariance = np.eye(6) * 0.05
        backend.add_loop_closure_factor(loop_i, loop_j, relative_pose, covariance)
        print(f"添加闭环因子: 节点 {loop_i} <-> 节点 {loop_j}")

    # 创建一个子图（模拟子图分割）
    backend.submaps.append({
        "start_id": 0,
        "end_id": 19,
        "nodes": list(range(20)),
        "features": []
    })

    backend.submaps.append({
        "start_id": 20,
        "end_id": 39,
        "nodes": list(range(20, 40)),
        "features": []
    })

    # 执行优化
    print("执行优化...")
    optimized_poses = backend.optimize()
    print(f"优化完成，优化了 {len(optimized_poses)} 个位姿")

    # 显示所有可视化
    print("显示可视化...")

    # 1. 2D轨迹图
    backend.visualize(show_2d=True, show_3d=False, show_errors=True,
                      save_prefix="factor_graph_demo")

    # 2. 3D轨迹图
    backend.visualize(show_2d=False, show_3d=True, show_errors=False,
                      save_prefix="factor_graph_demo")

    # 3. 统计信息
    backend.plot_statistics()

    print("演示完成！生成的图片已保存为 'factor_graph_demo_*.png'")


# 在文件末尾添加
if __name__ == "__main__":
    main()