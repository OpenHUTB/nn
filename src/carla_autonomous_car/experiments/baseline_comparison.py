import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class MonitorVisualizer:
    """可视化预处理后的monitor数据"""

    def __init__(self, data_path):
        """
        初始化
        Args:
            data_path: 处理后的数据文件路径
        """
        self.data_path = Path(data_path)
        self.df = None
        self.output_dir = Path('./visualizations')
        self.output_dir.mkdir(exist_ok=True)

    def load_data(self):
        """加载数据"""
        if self.data_path.suffix == '.parquet':
            self.df = pd.read_parquet(self.data_path)
        else:
            self.df = pd.read_csv(self.data_path)

        print(f"加载数据: {len(self.df)} 行, {len(self.df.columns)} 列")
        print(f"Agent数量: {self.df['agent_name'].nunique()}")

    def plot_learning_curves(self, metric='reward_ma100', figsize=(14, 8)):
        """绘制学习曲线（按agent分组）"""
        fig, axes = plt.subplots(2, 1, figsize=figsize)

        # 获取所有唯一的agent
        agents = sorted(self.df['agent_name'].unique())

        # 为不同的agent_type设置不同颜色
        agent_types = self.df['agent_type'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(agent_types)))
        type_color_map = dict(zip(agent_types, colors))

        # 子图1: 所有agent的学习曲线
        ax1 = axes[0]
        for agent in agents:
            agent_df = self.df[self.df['agent_name'] == agent]
            agent_type = agent_df['agent_type'].iloc[0]
            color = type_color_map[agent_type]

            ax1.plot(agent_df['training_step'],
                    agent_df[metric],
                    label=agent,
                    alpha=0.7,
                    linewidth=2,
                    color=color)

        ax1.set_xlabel('Training Steps', fontsize=12)
        ax1.set_ylabel('Reward (MA-100)', fontsize=12)
        ax1.set_title('Learning Curves - All Agents', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 子图2: 按agent_type分组平均
        ax2 = axes[1]
        for agent_type in agent_types:
            type_df = self.df[self.df['agent_type'] == agent_type]

            # 按training_step分组计算平均值
            grouped = type_df.groupby('training_step')[metric].agg(['mean', 'std'])

            color = type_color_map[agent_type]
            ax2.plot(grouped.index, grouped['mean'],
                    label=f'{agent_type} (mean)',
                    linewidth=2.5,
                    color=color)
            ax2.fill_between(grouped.index,
                            grouped['mean'] - grouped['std'],
                            grouped['mean'] + grouped['std'],
                            alpha=0.2,
                            color=color)

        ax2.set_xlabel('Training Steps', fontsize=12)
        ax2.set_ylabel('Reward (MA-100)', fontsize=12)
        ax2.set_title('Learning Curves - By Agent Type', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / 'learning_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
        plt.close()

    def plot_success_rate(self, figsize=(12, 6)):
        """绘制成功率对比"""
        fig, ax = plt.subplots(figsize=figsize)

        # 计算每个agent的成功率
        success_stats = self.df.groupby('agent_name').agg({
            'success': 'mean',
            'agent_type': 'first'
        }).reset_index()
        success_stats.columns = ['agent_name', 'success_rate', 'agent_type']
        success_stats = success_stats.sort_values('success_rate', ascending=False)

        # 设置颜色
        agent_types = success_stats['agent_type'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(agent_types)))
        type_color_map = dict(zip(agent_types, colors))
        bar_colors = [type_color_map[t] for t in success_stats['agent_type']]

        # 绘制柱状图
        bars = ax.bar(range(len(success_stats)),
                      success_stats['success_rate'],
                      color=bar_colors,
                      edgecolor='black',
                      linewidth=1.5)

        ax.set_xlabel('Agents', fontsize=12)
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_title('Success Rate Comparison Across Agents', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(success_stats)))
        ax.set_xticklabels(success_stats['agent_name'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for i, (bar, rate) in enumerate(zip(bars, success_stats['success_rate'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.2%}',
                   ha='center', va='bottom', fontsize=9)

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=type_color_map[t],
                                edgecolor='black',
                                label=t)
                          for t in agent_types]
        ax.legend(handles=legend_elements, title='Agent Type', fontsize=10)

        plt.tight_layout()
        save_path = self.output_dir / 'success_rate_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
        plt.close()

    def plot_reward_distribution(self, figsize=(14, 6)):
        """绘制奖励分布"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 子图1: 小提琴图
        ax1 = axes[0]
        agent_types = sorted(self.df['agent_type'].unique())

        data_to_plot = [self.df[self.df['agent_type'] == t]['reward'].values
                       for t in agent_types]

        parts = ax1.violinplot(data_to_plot,
                               positions=range(len(agent_types)),
                               showmeans=True,
                               showmedians=True)

        ax1.set_xlabel('Agent Type', fontsize=12)
        ax1.set_ylabel('Reward', fontsize=12)
        ax1.set_title('Reward Distribution by Agent Type', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(agent_types)))
        ax1.set_xticklabels(agent_types)
        ax1.grid(axis='y', alpha=0.3)

        # 子图2: 箱线图（按agent）
        ax2 = axes[1]
        agents = sorted(self.df['agent_name'].unique())

        boxplot_data = [self.df[self.df['agent_name'] == agent]['reward'].values
                       for agent in agents]

        bp = ax2.boxplot(boxplot_data,
                        labels=agents,
                        patch_artist=True,
                        showmeans=True)

        # 为不同agent_type设置不同颜色
        for i, agent in enumerate(agents):
            agent_type = self.df[self.df['agent_name'] == agent]['agent_type'].iloc[0]
            color_idx = list(agent_types).index(agent_type)
            bp['boxes'][i].set_facecolor(plt.cm.Set3(color_idx / len(agent_types)))

        ax2.set_xlabel('Agents', fontsize=12)
        ax2.set_ylabel('Reward', fontsize=12)
        ax2.set_title('Reward Distribution by Agent', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / 'reward_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
        plt.close()

    def plot_metrics_heatmap(self, figsize=(12, 8)):
        """绘制多指标热图"""
        # 选择关键指标
        metrics = ['reward', 'ego_speed', 'distance_traveled',
                  'collision', 'off_road', 'success']

        # 计算每个agent的平均值
        heatmap_data = self.df.groupby('agent_name')[metrics].mean()

        # 标准化数据（除了collision和off_road）
        normalized_data = heatmap_data.copy()
        for col in metrics:
            if col not in ['collision', 'off_road']:
                normalized_data[col] = (heatmap_data[col] - heatmap_data[col].min()) / \
                                       (heatmap_data[col].max() - heatmap_data[col].min())

        # 绘制热图
        plt.figure(figsize=figsize)
        sns.heatmap(normalized_data.T,
                   annot=True,
                   fmt='.3f',
                   cmap='RdYlGn',
                   center=0.5,
                   cbar_kws={'label': 'Normalized Value'},
                   linewidths=0.5)

        plt.xlabel('Agents', fontsize=12)
        plt.ylabel('Metrics', fontsize=12)
        plt.title('Performance Metrics Heatmap (Normalized)',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = self.output_dir / 'metrics_heatmap.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
        plt.close()

    def plot_episode_length_analysis(self, figsize=(14, 6)):
        """分析episode长度"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 子图1: episode长度随训练步数变化
        ax1 = axes[0]
        for agent in sorted(self.df['agent_name'].unique()):
            agent_df = self.df[self.df['agent_name'] == agent]

            # 计算滑动平均
            agent_df_sorted = agent_df.sort_values('training_step')
            ma = agent_df_sorted['length'].rolling(window=50, min_periods=1).mean()

            ax1.plot(agent_df_sorted['training_step'],
                    ma,
                    label=agent,
                    alpha=0.7)

        ax1.set_xlabel('Training Steps', fontsize=12)
        ax1.set_ylabel('Episode Length (MA-50)', fontsize=12)
        ax1.set_title('Episode Length Over Training', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 子图2: 平均episode长度对比
        ax2 = axes[1]
        length_stats = self.df.groupby('agent_name')['length'].mean().sort_values()

        bars = ax2.barh(range(len(length_stats)),
                       length_stats.values,
                       color=plt.cm.viridis(np.linspace(0, 1, len(length_stats))))

        ax2.set_yticks(range(len(length_stats)))
        ax2.set_yticklabels(length_stats.index)
        ax2.set_xlabel('Average Episode Length', fontsize=12)
        ax2.set_title('Average Episode Length by Agent', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, length_stats.values)):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{val:.1f}',
                    ha='left', va='center', fontsize=9)

        plt.tight_layout()
        save_path = self.output_dir / 'episode_length_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
        plt.close()

    def generate_all_visualizations(self):
        """生成所有可视化图表"""
        print("\n开始生成可视化图表...")
        print("="*60)

        self.plot_learning_curves()
        self.plot_success_rate()
        self.plot_reward_distribution()
        self.plot_metrics_heatmap()
        self.plot_episode_length_analysis()

        print("="*60)
        print(f"\n所有图表已保存至: {self.output_dir}/")
        print("\n生成的图表:")
        for img in sorted(self.output_dir.glob('*.png')):
            print(f"  - {img.name}")


def main():
    """主函数"""
    # 数据路径（可以是CSV或Parquet）
    data_path = './processed_data/all_monitors_combined.parquet'
    # 或使用: data_path = './processed_data/all_monitors_combined.csv'

    # 创建可视化器
    visualizer = MonitorVisualizer(data_path)

    # 加载数据
    visualizer.load_data()

    # 生成所有可视化
    visualizer.generate_all_visualizations()

    print("\n可视化完成！")


if __name__ == "__main__":
    main()