"""
analyze_results.py - 分析实验结果

功能：
1. 读取所有monitor.csv文件
2. 计算关键指标统计
3. 生成对比表格
4. 保存为CSV和LaTeX格式
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json


class ExperimentAnalyzer:
    def __init__(self, base_dir='experiments/results'):
        self.base_dir = Path(base_dir)
        self.methods = ['baseline', 'improved', 'lyapunov']
        self.results = {}

    def load_data(self):
        """加载所有实验数据"""
        print("📂 Loading experiment data...")

        for method in self.methods:
            method_dir = self.base_dir / method / 'logs'

            if not method_dir.exists():
                print(f"⚠️  Warning: {method_dir} not found, skipping {method}")
                continue

            # 查找所有monitor.csv文件
            csv_files = list(method_dir.glob('monitor_seed*.csv'))

            if not csv_files:
                print(f"⚠️  Warning: No monitor files found for {method}")
                continue

            # 读取所有种子的数据
            dfs = []
            for csv_file in csv_files:
                try:
                    # 跳过第一行（注释）
                    df = pd.read_csv(csv_file, skiprows=1)
                    seed = int(csv_file.stem.split('seed')[-1])
                    df['seed'] = seed
                    dfs.append(df)
                    print(f"  ✓ Loaded {csv_file.name}")
                except Exception as e:
                    print(f"  ✗ Error loading {csv_file.name}: {e}")

            if dfs:
                self.results[method] = pd.concat(dfs, ignore_index=True)
                print(f"✅ {method}: {len(csv_files)} seeds, {len(self.results[method])} episodes")

        print("")

    def compute_statistics(self):
        """计算统计指标"""
        print("📊 Computing statistics...")

        stats = {}

        for method, df in self.results.items():
            print(f"\n🔍 Analyzing {method.upper()}:")

            method_stats = {}

            # 基础指标
            method_stats['total_episodes'] = len(df)
            method_stats['total_timesteps'] = df['l'].sum()

            # 奖励指标
            method_stats['mean_reward'] = df['r'].mean()
            method_stats['std_reward'] = df['r'].std()
            method_stats['max_reward'] = df['r'].max()
            method_stats['min_reward'] = df['r'].min()

            # 成功率（track_finished = True）
            if 'track_finished' in df.columns:
                method_stats['success_rate'] = (df['track_finished'] > 0).mean()
            else:
                method_stats['success_rate'] = np.nan

            # 碰撞率
            if 'collision' in df.columns:
                method_stats['collision_rate'] = (df['collision'] > 0).mean()
            else:
                method_stats['collision_rate'] = np.nan

            # 换道成功率
            if 'ep_succ_lane_changes' in df.columns:
                method_stats['avg_lane_changes'] = df['ep_succ_lane_changes'].mean()
            else:
                method_stats['avg_lane_changes'] = np.nan

            # 安全违规
            if 'ep_safety_viols' in df.columns:
                method_stats['avg_safety_viols'] = df['ep_safety_viols'].mean()
            else:
                method_stats['avg_safety_viols'] = np.nan

            # 行驶距离
            if 'distance_traveled' in df.columns:
                method_stats['avg_distance'] = df['distance_traveled'].mean()
            else:
                method_stats['avg_distance'] = np.nan

            # 最后100 episodes的平均奖励（学习后期性能）
            method_stats['final_100_mean_reward'] = df['r'].tail(100).mean()

            stats[method] = method_stats

            # 打印统计
            print(f"  Episodes: {method_stats['total_episodes']}")
            print(f"  Mean Reward: {method_stats['mean_reward']:.2f} ± {method_stats['std_reward']:.2f}")
            print(f"  Final 100 Episodes: {method_stats['final_100_mean_reward']:.2f}")
            print(f"  Success Rate: {method_stats['success_rate'] * 100:.1f}%")
            print(f"  Collision Rate: {method_stats['collision_rate'] * 100:.1f}%")
            print(f"  Avg Lane Changes: {method_stats['avg_lane_changes']:.2f}")

        return stats

    def create_comparison_table(self, stats):
        """创建对比表格"""
        print("\n📋 Creating comparison table...")

        # 创建DataFrame
        rows = []
        for method, method_stats in stats.items():
            rows.append({
                'Method': method.capitalize(),
                'Mean Reward': f"{method_stats['mean_reward']:.2f} ± {method_stats['std_reward']:.2f}",
                'Final 100 Reward': f"{method_stats['final_100_mean_reward']:.2f}",
                'Success Rate (%)': f"{method_stats['success_rate'] * 100:.1f}",
                'Collision Rate (%)': f"{method_stats['collision_rate'] * 100:.1f}",
                'Avg Lane Changes': f"{method_stats['avg_lane_changes']:.2f}",
                'Avg Distance (m)': f"{method_stats['avg_distance']:.1f}",
            })

        df_comparison = pd.DataFrame(rows)

        # 保存CSV
        output_file = self.base_dir.parent / 'analysis' / 'comparison_table.csv'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df_comparison.to_csv(output_file, index=False)
        print(f"✅ Saved to {output_file}")

        # 保存LaTeX
        latex_file = self.base_dir.parent / 'analysis' / 'comparison_table.tex'
        df_comparison.to_latex(latex_file, index=False)
        print(f"✅ Saved LaTeX to {latex_file}")

        # 打印表格
        print("\n" + "=" * 80)
        print(df_comparison.to_string(index=False))
        print("=" * 80)

        return df_comparison

    def save_statistics(self, stats):
        """保存统计数据为JSON"""
        output_file = self.base_dir.parent / 'analysis' / 'statistics.json'
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n✅ Statistics saved to {output_file}")

    def run_analysis(self):
        """运行完整分析"""
        self.load_data()

        if not self.results:
            print("❌ No data loaded, cannot perform analysis")
            return

        stats = self.compute_statistics()
        self.create_comparison_table(stats)
        self.save_statistics(stats)

        print("\n🎉 Analysis complete!")
        print(f"📁 Results saved to {self.base_dir.parent / 'analysis'}")


if __name__ == '__main__':
    analyzer = ExperimentAnalyzer()
    analyzer.run_analysis()