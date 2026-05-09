"""
plot_results.py - 绘制实验结果

生成图表：
1. 学习曲线（平均奖励 vs 时间步）
2. 最终性能对比（柱状图）
3. 成功率/碰撞率对比
4. 换道成功率对比
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.ndimage import uniform_filter1d

# 设置绘图样式
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12


class ResultPlotter:
    def __init__(self, base_dir='experiments/results'):
        self.base_dir = Path(base_dir)
        self.figure_dir = self.base_dir.parent / 'figures'
        self.figure_dir.mkdir(parents=True, exist_ok=True)

        self.methods = ['baseline', 'improved', 'lyapunov']
        self.method_colors = {
            'baseline': '#E74C3C',  # 红色
            'improved': '#3498DB',  # 蓝色
            'lyapunov': '#2ECC71',  # 绿色
        }
        self.method_labels = {
            'baseline': 'Baseline',
            'improved': 'Improved (Simple Shaping)',
            'lyapunov': 'Lyapunov (Theory-based)',
        }

        self.data = {}

    def load_data(self):
        """加载数据"""
        print("📂 Loading data for plotting...")

        for method in self.methods:
            method_dir = self.base_dir / method / 'logs'

            if not method_dir.exists():
                continue

            csv_files = list(method_dir.glob('monitor_seed*.csv'))

            if not csv_files:
                continue

            dfs = []
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file, skiprows=1)
                    seed = int(csv_file.stem.split('seed')[-1])
                    df['seed'] = seed
                    df['episode'] = range(len(df))
                    dfs.append(df)
                except Exception as e:
                    print(f"Error loading {csv_file}: {e}")

            if dfs:
                self.data[method] = pd.concat(dfs, ignore_index=True)
                print(f"  ✓ {method}: {len(csv_files)} seeds")

        print("")

    def plot_learning_curves(self):
        """绘制学习曲线"""
        print("📈 Plotting learning curves...")

        fig, ax = plt.subplots(figsize=(14, 7))

        for method, df in self.data.items():
            # 按seed分组，计算每个episode的平均奖励
            grouped = df.groupby('episode')['r'].agg(['mean', 'std', 'count'])

            # 计算标准误
            grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])

            # 平滑
            window_size = 50
            smoothed_mean = uniform_filter1d(grouped['mean'], size=window_size, mode='nearest')
            smoothed_se = uniform_filter1d(grouped['se'], size=window_size, mode='nearest')

            episodes = grouped.index.values

            # 绘制均值线
            ax.plot(episodes, smoothed_mean,
                    label=self.method_labels[method],
                    color=self.method_colors[method],
                    linewidth=2)

            # 绘制置信区间
            ax.fill_between(episodes,
                            smoothed_mean - smoothed_se,
                            smoothed_mean + smoothed_se,
                            color=self.method_colors[method],
                            alpha=0.2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Reward')
        ax.set_title('Learning Curves (Mean ± SE, smoothed over 50 episodes)')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        # 保存
        output_file = self.figure_dir / 'learning_curves' / 'reward_vs_episode.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to {output_file}")
        plt.close()

    def plot_final_performance(self):
        """绘制最终性能对比"""
        print("📊 Plotting final performance comparison...")

        # 计算每个方法最后100 episodes的平均奖励
        final_rewards = {}
        for method, df in self.data.items():
            # 每个seed的最后100 episodes
            seed_rewards = []
            for seed in df['seed'].unique():
                seed_df = df[df['seed'] == seed]
                last_100 = seed_df['r'].tail(100).mean()
                seed_rewards.append(last_100)

            final_rewards[method] = {
                'mean': np.mean(seed_rewards),
                'std': np.std(seed_rewards),
                'values': seed_rewards
            }

        # 绘图
        fig, ax = plt.subplots(figsize=(10, 6))

        methods_ordered = ['baseline', 'improved', 'lyapunov']
        x_pos = np.arange(len(methods_ordered))
        means = [final_rewards[m]['mean'] for m in methods_ordered]
        stds = [final_rewards[m]['std'] for m in methods_ordered]
        colors = [self.method_colors[m] for m in methods_ordered]
        labels = [self.method_labels[m] for m in methods_ordered]

        bars = ax.bar(x_pos, means, yerr=stds, capsize=10,
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        # 添加数值标签
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.5, f'{mean:.2f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylabel('Average Reward (last 100 episodes)')
        ax.set_title('Final Performance Comparison')
        ax.grid(axis='y', alpha=0.3)

        # 保存
        output_file = self.figure_dir / 'comparisons' / 'final_performance.png'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to {output_file}")
        plt.close()

    def plot_success_collision_rates(self):
        """绘制成功率和碰撞率对比"""
        print("📊 Plotting success and collision rates...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        methods_ordered = ['baseline', 'improved', 'lyapunov']
        x_pos = np.arange(len(methods_ordered))

        # 成功率
        success_rates = []
        collision_rates = []

        for method in methods_ordered:
            df = self.data[method]

            if 'track_finished' in df.columns:
                success_rate = (df['track_finished'] > 0).mean() * 100
            else:
                success_rate = 0
            success_rates.append(success_rate)

            if 'collision' in df.columns:
                collision_rate = (df['collision'] > 0).mean() * 100
            else:
                collision_rate = 0
            collision_rates.append(collision_rate)

        colors = [self.method_colors[m] for m in methods_ordered]
        labels = [self.method_labels[m] for m in methods_ordered]

        # 成功率
        bars1 = ax1.bar(x_pos, success_rates, color=colors, alpha=0.8,
                        edgecolor='black', linewidth=1.5)
        for i, rate in enumerate(success_rates):
            ax1.text(i, rate + 1, f'{rate:.1f}%',
                     ha='center', va='bottom', fontweight='bold')

        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels, rotation=15, ha='right')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Track Completion Rate')
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', alpha=0.3)

        # 碰撞率
        bars2 = ax2.bar(x_pos, collision_rates, color=colors, alpha=0.8,
                        edgecolor='black', linewidth=1.5)
        for i, rate in enumerate(collision_rates):
            ax2.text(i, rate + 1, f'{rate:.1f}%',
                     ha='center', va='bottom', fontweight='bold')

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, rotation=15, ha='right')
        ax2.set_ylabel('Collision Rate (%)')
        ax2.set_title('Collision Rate')
        ax2.set_ylim(0, max(collision_rates) * 1.2 if collision_rates else 50)
        ax2.grid(axis='y', alpha=0.3)

        # 保存
        output_file = self.figure_dir / 'comparisons' / 'success_collision_rates.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to {output_file}")
        plt.close()

    def plot_lane_change_success(self):
        """绘制换道成功率"""
        print("📊 Plotting lane change statistics...")

        fig, ax = plt.subplots(figsize=(10, 6))

        methods_ordered = ['baseline', 'improved', 'lyapunov']
        x_pos = np.arange(len(methods_ordered))

        avg_lane_changes = []
        for method in methods_ordered:
            df = self.data[method]
            if 'ep_succ_lane_changes' in df.columns:
                avg_lc = df['ep_succ_lane_changes'].mean()
            else:
                avg_lc = 0
            avg_lane_changes.append(avg_lc)

        colors = [self.method_colors[m] for m in methods_ordered]
        labels = [self.method_labels[m] for m in methods_ordered]

        bars = ax.bar(x_pos, avg_lane_changes, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=1.5)

        for i, avg in enumerate(avg_lane_changes):
            ax.text(i, avg + 0.05, f'{avg:.2f}',
                    ha='center', va='bottom', fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylabel('Average Successful Lane Changes per Episode')
        ax.set_title('Lane Change Performance')
        ax.grid(axis='y', alpha=0.3)

        # 保存
        output_file = self.figure_dir / 'comparisons' / 'lane_change_success.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to {output_file}")
        plt.close()

    def generate_all_plots(self):
        """生成所有图表"""
        self.load_data()

        if not self.data:
            print("❌ No data loaded")
            return

        self.plot_learning_curves()
        self.plot_final_performance()
        self.plot_success_collision_rates()
        self.plot_lane_change_success()

        print(f"\n🎉 All plots generated!")
        print(f"📁 Saved to {self.figure_dir}")


if __name__ == '__main__':
    plotter = ResultPlotter()
    plotter.generate_all_plots()