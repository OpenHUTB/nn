"""
可视化工具模块

提供训练曲线绘制、Q表可视化、策略可视化等功能
"""

import os
from typing import Optional, List, Dict

import matplotlib.pyplot as plt
import numpy as np


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
