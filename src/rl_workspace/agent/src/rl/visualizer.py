"""
可视化工具模块

提供训练曲线绘制、Q表可视化、策略可视化等功能
扩展：实时训练曲线、策略热力图、价值函数曲面图
"""

import os
from typing import Optional, List, Dict, Callable

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # 用于3D曲面


class LivePlot:
    """
    实时更新训练曲线，支持多条曲线
    用法：
        live = LivePlot(xlabel='Episode', ylabel='Reward')
        live.add_curve('reward', color='red')
        for episode in range(N):
            live.update('reward', episode, episode_reward)
    """

    def __init__(self, xlabel: str = 'Episode', ylabel: str = 'Value', title: str = 'Live Plot', max_points: int = 500):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.ax.set_xlabel(xlabel, fontsize=12)
        self.ax.set_ylabel(ylabel, fontsize=12)
        self.ax.set_title(title, fontsize=14)
        self.ax.grid(True, alpha=0.3)
        self.lines: Dict[str, plt.Line2D] = {}
        self.data: Dict[str, tuple] = {}  # (xs, ys)
        self.max_points = max_points

    def add_curve(self, name: str, color: Optional[str] = None):
        """添加一条新的曲线"""
        line, = self.ax.plot([], [], label=name, color=color, linewidth=2)
        self.lines[name] = line
        self.data[name] = ([], [])
        self.ax.legend(loc='best')

    def update(self, name: str, x: float, y: float):
        """更新指定曲线的最新点"""
        if name not in self.lines:
            self.add_curve(name)
        xs, ys = self.data[name]
        xs.append(x)
        ys.append(y)
        if len(xs) > self.max_points:
            xs.pop(0)
            ys.pop(0)
        self.lines[name].set_data(xs, ys)
        # 自动调整坐标轴范围
        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(0.001)  # 短暂暂停以刷新图形

    def save(self, filepath: str, dpi: int = 150):
        """保存当前图表（自动创建目录）"""
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        self.fig.savefig(filepath, dpi=dpi)
        print(f"实时图表已保存到: {filepath}")

    def close(self):
        plt.close(self.fig)


class PolicyHeatmap:
    """
    离散动作策略热力图，实时展示各动作的概率分布
    需要算法提供 get_action_probs(state) 方法返回概率数组 (num_actions,)
    """

    def __init__(self, num_actions: int, env_name: str = "Unknown"):
        self.num_actions = num_actions
        self.fig, self.ax = plt.subplots(figsize=(8, 3))
        self.im = None
        self.env_name = env_name
        self.step = 0

    def update(self, probs: np.ndarray, step: int):
        """
        probs: shape (num_actions,)，概率值，应归一化
        step: 当前训练步数/episode数
        """
        self.step = step
        if self.im is None:
            # 初始绘制
            self.im = self.ax.imshow([probs], aspect='auto', cmap='viridis', vmin=0, vmax=1)
            self.ax.set_ylabel('Policy', fontsize=12)
            self.ax.set_xlabel('Action', fontsize=12)
            self.ax.set_title(f'Policy Probabilities at Step {step} ({self.env_name})', fontsize=14)
            self.ax.set_yticks([])
            self.ax.set_xticks(range(self.num_actions))
            plt.colorbar(self.im, ax=self.ax, label='Probability')
        else:
            self.im.set_array([probs])
            self.ax.set_title(f'Policy Probabilities at Step {step} ({self.env_name})')
        plt.pause(0.001)

    def save(self, filepath: str):
        """保存当前图表（自动创建目录）"""
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        self.fig.savefig(filepath, dpi=150)
        print(f"策略热力图已保存到: {filepath}")

    def close(self):
        plt.close(self.fig)


class ValueSurface:
    """
    3D价值函数曲面图，适用于2维连续状态空间（如位置+速度）
    需要提供一个 value_function: np.ndarray -> float 的可调用对象
    """

    def __init__(self, bounds: tuple = ((-3, 3), (-3, 3)), resolution: int = 30):
        """
        bounds: ((x_min, x_max), (y_min, y_max))
        resolution: 网格点数
        """
        self.bounds = bounds
        self.res = resolution
        x = np.linspace(bounds[0][0], bounds[0][1], resolution)
        y = np.linspace(bounds[1][0], bounds[1][1], resolution)
        self.xx, self.yy = np.meshgrid(x, y)
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

    def update(self, value_func: Callable[[np.ndarray], float], step: int):
        """
        value_func: 接受一个形状 (2,) 的状态数组，返回标量价值
        step: 当前训练步数/episode数
        """
        zz = np.zeros_like(self.xx)
        for i in range(self.res):
            for j in range(self.res):
                state = np.array([self.xx[i, j], self.yy[i, j]])
                zz[i, j] = value_func(state)
        self.ax.clear()
        self.ax.plot_surface(self.xx, self.yy, zz, cmap='coolwarm', alpha=0.8)
        self.ax.set_xlabel('State dim 1', fontsize=10)
        self.ax.set_ylabel('State dim 2', fontsize=10)
        self.ax.set_zlabel('Value', fontsize=10)
        self.ax.set_title(f'Value Function Surface at Step {step}')
        plt.pause(0.001)

    def save(self, filepath: str):
        """保存当前图表（自动创建目录）"""
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        self.fig.savefig(filepath, dpi=150)
        print(f"价值曲面图已保存到: {filepath}")

    def close(self):
        plt.close(self.fig)


# ========== 以下是原有代码（TrainingVisualizer, QTableVisualizer, PerformanceAnalyzer）保持不变 ==========

class TrainingVisualizer:
    """训练可视化器"""
    
    @staticmethod
    def plot_training_curve(
        rewards: List[float],
        title: str = "训练曲线",
        xlabel: str = "Episode",
        ylabel: str = "Reward",
        window_size: int = 100,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """绘制训练曲线"""
        plt.figure(figsize=(12, 6))
        
        # 原始奖励曲线
        plt.plot(rewards, label='每轮奖励', alpha=0.5, linewidth=1)
        
        # 滑动平均曲线
        if len(rewards) >= window_size:
            running_mean = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(
                range(window_size-1, len(rewards)),
                running_mean,
                label=f'{window_size}轮滑动平均',
                color='red',
                linewidth=2
            )
        
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"训练曲线已保存到: {save_path}")
        
        if show:
            plt.show()
    
    @staticmethod
    def plot_multiple_curves(
        curves: Dict[str, List[float]],
        title: str = "训练曲线对比",
        xlabel: str = "Episode",
        ylabel: str = "Reward",
        window_size: int = 100,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """绘制多条训练曲线进行对比"""
        plt.figure(figsize=(12, 6))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan']
        
        for i, (name, rewards) in enumerate(curves.items()):
            # 原始曲线
            plt.plot(rewards, label=name, alpha=0.3, linewidth=1, color=colors[i % len(colors)])
            
            # 滑动平均曲线
            if len(rewards) >= window_size:
                running_mean = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                plt.plot(
                    range(window_size-1, len(rewards)),
                    running_mean,
                    label=f'{name} (平均)',
                    color=colors[i % len(colors)],
                    linewidth=2
                )
        
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"对比曲线已保存到: {save_path}")
        
        if show:
            plt.show()


class QTableVisualizer:
    """Q表可视化器"""
    
    @staticmethod
    def visualize_q_table(
        q_table: np.ndarray,
        title: str = "Q表可视化",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """可视化Q表为热力图"""
        plt.figure(figsize=(10, 8))
        
        # 如果是一维状态空间
        if q_table.ndim == 2:
            im = plt.imshow(q_table, cmap='viridis', interpolation='nearest')
            plt.colorbar(im, label='Q值')
            
            # 添加数值标签
            for i in range(q_table.shape[0]):
                for j in range(q_table.shape[1]):
                    plt.text(j, i, f'{q_table[i, j]:.2f}',
                             ha='center', va='center', color='white', fontsize=8)
            
            plt.xlabel('动作', fontsize=12)
            plt.ylabel('状态', fontsize=12)
        
        plt.title(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"Q表可视化已保存到: {save_path}")
        
        if show:
            plt.show()
    
    @staticmethod
    def visualize_policy(
        q_table: np.ndarray,
        grid_size: int = 4,
        title: str = "策略可视化",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """可视化策略（适用于网格环境）"""
        plt.figure(figsize=(grid_size * 2, grid_size * 2))
        
        # 获取最优动作
        policy = np.argmax(q_table, axis=1)
        
        # 动作映射
        actions = ['←', '↓', '→', '↑']
        
        # 绘制网格
        for i in range(grid_size):
            for j in range(grid_size):
                state = i * grid_size + j
                action = actions[policy[state]]
                
                # 绘制单元格
                plt.text(j + 0.5, grid_size - i - 0.5, action,
                         ha='center', va='center', fontsize=24, fontweight='bold')
        
        # 设置网格
        plt.xlim(0, grid_size)
        plt.ylim(0, grid_size)
        plt.grid(True, linewidth=2)
        
        plt.title(title, fontsize=14)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"策略可视化已保存到: {save_path}")
        
        if show:
            plt.show()


class PerformanceAnalyzer:
    """性能分析器"""
    
    @staticmethod
    def plot_success_rate(
        rewards: List[float],
        window_size: int = 100,
        title: str = "成功率变化",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """绘制成功率变化曲线"""
        # 计算每window_size轮的成功率
        success_rates = []
        for i in range(len(rewards) - window_size + 1):
            window = rewards[i:i+window_size]
            success_rate = sum(1 for r in window if r > 0) / window_size * 100
            success_rates.append(success_rate)
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(window_size-1, len(rewards)), success_rates, label='成功率', color='green')
        
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('成功率 (%)', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"成功率曲线已保存到: {save_path}")
        
        if show:
            plt.show()