import pandas as pd
import numpy as np
import os
from pathlib import Path
import re
import json
import warnings

warnings.filterwarnings('ignore')


class MonitorPreprocessor:
    """
    预处理多个agent文件夹中的monitor.csv文件 - 可指定文件夹版本
    """

    def __init__(self, root_dir, target_agents=None):
        """
        初始化
        Args:
            root_dir: 包含所有agent文件夹的根目录
            target_agents: 要处理的agent列表，None表示处理所有agent
                          示例: ['agent_101_improved', 'agent_102_improved']
        """
        self.root_dir = Path(root_dir)
        self.target_agents = target_agents
        self.all_data = []
        self.failed_files = []
        self.column_inconsistencies = []

    def find_monitor_files(self):
        """查找指定agent文件夹中的monitor.csv文件"""
        monitor_files = []

        # 查找所有agent文件夹
        all_agent_folders = [f for f in self.root_dir.iterdir()
                             if f.is_dir() and 'agent' in f.name.lower()]

        # 如果指定了target_agents，则只处理这些文件夹
        if self.target_agents:
            agent_folders = [f for f in all_agent_folders
                             if f.name in self.target_agents]
            print(f"\n📌 指定处理 {len(self.target_agents)} 个agent:")
            for agent in self.target_agents:
                print(f"  - {agent}")

            # 检查是否有指定的文件夹不存在
            found_names = {f.name for f in agent_folders}
            missing = set(self.target_agents) - found_names
            if missing:
                print(f"\n⚠️  以下指定的agent文件夹不存在:")
                for agent in missing:
                    print(f"  - {agent}")
        else:
            agent_folders = all_agent_folders
            print(f"\n📌 处理所有agent文件夹 ({len(agent_folders)} 个)")

        print(f"\n找到的文件:")
        for folder in sorted(agent_folders):
            monitor_path = folder / 'monitor.csv'
            if monitor_path.exists():
                monitor_files.append({
                    'path': monitor_path,
                    'agent_name': folder.name
                })
                print(f"  ✓ {monitor_path}")
            else:
                print(f"  ✗ {folder.name}/monitor.csv 不存在")

        return monitor_files

    def extract_metadata(self, file_path):
        """从monitor.csv第一行提取元数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()

            if first_line.startswith('#'):
                try:
                    metadata = json.loads(first_line[1:])
                    return metadata
                except:
                    return {}
        except:
            return {}
        return {}

    def parse_agent_info(self, agent_name):
        """从agent文件夹名称提取信息"""
        match = re.match(r'agent_(\d+)_(\w+)', agent_name)
        if match:
            return {
                'agent_id': int(match.group(1)),
                'agent_type': match.group(2),
                'agent_series': int(match.group(1)) // 100
            }

        match = re.match(r'agent_(\d+)', agent_name)
        if match:
            return {
                'agent_id': int(match.group(1)),
                'agent_type': 'baseline',
                'agent_series': 0
            }

        return {'agent_id': None, 'agent_type': 'unknown', 'agent_series': None}

    def detect_column_count(self, file_path, sample_lines=100):
        """检测CSV文件的列数"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = []
                for i, line in enumerate(f):
                    if i == 0 or not line.strip():
                        continue
                    if i > sample_lines:
                        break
                    lines.append(line.count(',') + 1)

                if not lines:
                    return None, None

                from collections import Counter
                counter = Counter(lines)
                most_common_count = counter.most_common(1)[0][0]

                if len(counter) > 1:
                    return most_common_count, dict(counter)

                return most_common_count, None
        except:
            return None, None

    def load_single_monitor_robust(self, file_info):
        """健壮地加载单个monitor.csv文件"""
        file_path = file_info['path']
        agent_name = file_info['agent_name']

        print(f"\n处理: {agent_name}")

        metadata = self.extract_metadata(file_path)

        col_info = self.detect_column_count(file_path)
        if col_info[0] is not None:
            expected_cols, inconsistencies = col_info
            if inconsistencies:
                print(f"  ⚠️  警告: 检测到列数不一致: {inconsistencies}")
                self.column_inconsistencies.append({
                    'agent': agent_name,
                    'inconsistencies': inconsistencies
                })

        df = None
        errors = []

        # 方法1: 标准读取
        try:
            df = pd.read_csv(file_path, skiprows=2, on_bad_lines='skip',
                             encoding='utf-8', engine='python')
            print(f"  ✓ 方法1成功: 加载 {len(df)} 行数据")
        except Exception as e:
            errors.append(f"方法1失败: {str(e)[:100]}")

        # 方法2: 使用comment参数
        if df is None:
            try:
                df = pd.read_csv(file_path, comment='#', skip_blank_lines=True,
                                 on_bad_lines='skip', encoding='utf-8', engine='python')
                print(f"  ✓ 方法2成功: 加载 {len(df)} 行数据")
            except Exception as e:
                errors.append(f"方法2失败: {str(e)[:100]}")

        # 方法3: 手动解析
        if df is None:
            try:
                df = self.manual_parse_csv(file_path)
                if df is not None:
                    print(f"  ✓ 方法3成功: 加载 {len(df)} 行数据")
            except Exception as e:
                errors.append(f"方法3失败: {str(e)[:100]}")

        # 方法4: 宽松模式
        if df is None:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i == 0 and line.startswith('#'):
                            continue
                        if line.strip() and not line.startswith('#'):
                            header_line = line.strip()
                            break

                headers = header_line.split(',')

                data_rows = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    skip_count = 0
                    for i, line in enumerate(f):
                        if i < 2:
                            continue
                        if not line.strip():
                            continue

                        parts = line.strip().split(',')
                        if len(parts) == len(headers):
                            data_rows.append(parts)
                        else:
                            skip_count += 1

                if data_rows:
                    df = pd.DataFrame(data_rows, columns=headers)
                    for col in df.columns:
                        try:
                            df[col] = pd.to_numeric(df[col])
                        except:
                            pass
                    print(f"  ✓ 方法4成功: 加载 {len(df)} 行数据 (跳过 {skip_count} 行格式错误)")
            except Exception as e:
                errors.append(f"方法4失败: {str(e)[:100]}")

        if df is None:
            print(f"  ✗ 所有方法都失败了:")
            for error in errors:
                print(f"    - {error}")
            self.failed_files.append({
                'agent': agent_name,
                'path': str(file_path),
                'errors': errors
            })
            return None

        # 添加agent信息
        df['agent_name'] = agent_name
        agent_info = self.parse_agent_info(agent_name)
        df['agent_id'] = agent_info['agent_id']
        df['agent_type'] = agent_info['agent_type']
        df['agent_series'] = agent_info['agent_series']

        if metadata:
            df['t_start'] = metadata.get('t_start', None)
            df['env_id'] = metadata.get('env_id', None)

        return df

    def manual_parse_csv(self, file_path):
        """手动解析CSV文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        header_idx = None
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith('#'):
                header_idx = i
                break

        if header_idx is None:
            return None

        headers = lines[header_idx].strip().split(',')
        expected_cols = len(headers)

        data_rows = []
        for line in lines[header_idx + 1:]:
            if not line.strip():
                continue

            parts = line.strip().split(',')
            if len(parts) == expected_cols:
                data_rows.append(parts)

        if not data_rows:
            return None

        df = pd.DataFrame(data_rows, columns=headers)

        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

        return df

    def load_all_monitors(self):
        """加载所有monitor.csv文件"""
        monitor_files = self.find_monitor_files()

        if not monitor_files:
            print("\n❌ 未找到任何monitor.csv文件！")
            return None

        print(f"\n总共找到 {len(monitor_files)} 个monitor.csv文件\n")
        print("=" * 70)

        for file_info in monitor_files:
            df = self.load_single_monitor_robust(file_info)
            if df is not None:
                self.all_data.append(df)

        if not self.all_data:
            print("\n✗ 没有成功加载任何数据!")
            return None

        print("\n" + "=" * 70)
        print("合并数据...")

        all_columns = [set(df.columns) for df in self.all_data]
        common_columns = set.intersection(*all_columns)

        if len(common_columns) < len(all_columns[0]):
            print(f"⚠️  警告: 检测到列不一致，使用公共列 ({len(common_columns)} 列)")
            self.all_data = [df[list(common_columns)] for df in self.all_data]

        combined_df = pd.concat(self.all_data, ignore_index=True)
        print(f"✓ 合并后总共 {len(combined_df)} 行数据")

        if self.failed_files:
            print(f"\n⚠️  {len(self.failed_files)} 个文件加载失败:")
            for failed in self.failed_files:
                print(f"  - {failed['agent']}")

        if self.column_inconsistencies:
            print(f"\n⚠️  {len(self.column_inconsistencies)} 个文件存在列数不一致:")
            for inc in self.column_inconsistencies:
                print(f"  - {inc['agent']}: {inc['inconsistencies']}")

        return combined_df

    def clean_data(self, df):
        """数据清洗"""
        print("\n" + "=" * 70)
        print("开始数据清洗...")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        numeric_cols = ['reward', 'length', 'training_step', 'ego_speed',
                        'distance_traveled', 'collision', 'off_road']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['episode_num'] = df.groupby('agent_name').cumcount()
        df['global_episode'] = range(len(df))

        if 'reward' in df.columns:
            df['cumulative_reward'] = df.groupby('agent_name')['reward'].cumsum()
        if 'training_step' in df.columns:
            df['cumulative_steps'] = df.groupby('agent_name')['training_step'].cummax()

        if 'reward' in df.columns:
            df['reward_ma100'] = df.groupby('agent_name')['reward'].transform(
                lambda x: x.rolling(window=100, min_periods=1).mean()
            )

        if all(col in df.columns for col in ['collision', 'off_road']):
            if 'track_finished' in df.columns:
                df['success'] = (~df['collision'].astype(bool)) & \
                                (~df['off_road'].astype(bool)) & \
                                (df['track_finished'].astype(bool))
            else:
                df['success'] = (~df['collision'].astype(bool)) & \
                                (~df['off_road'].astype(bool))

            df['success_rate_ma100'] = df.groupby('agent_name')['success'].transform(
                lambda x: x.rolling(window=100, min_periods=1).mean()
            )

        if 'reward' in df.columns:
            reward_threshold = df['reward'].quantile(0.01)
            df['is_outlier'] = df['reward'] < reward_threshold

        print("✓ 数据清洗完成")
        return df

    def generate_summary_stats(self, df):
        """生成汇总统计"""
        print("\n生成汇总统计...")

        agg_dict = {}

        if 'reward' in df.columns:
            agg_dict['reward'] = ['mean', 'std', 'min', 'max', 'median']
        if 'length' in df.columns:
            agg_dict['length'] = ['mean', 'sum']
        if 'training_step' in df.columns:
            agg_dict['training_step'] = 'max'
        if 'collision' in df.columns:
            agg_dict['collision'] = 'sum'
        if 'off_road' in df.columns:
            agg_dict['off_road'] = 'sum'
        if 'track_finished' in df.columns:
            agg_dict['track_finished'] = 'sum'
        if 'success' in df.columns:
            agg_dict['success'] = ['sum', 'mean']
        if 'ego_speed' in df.columns:
            agg_dict['ego_speed'] = 'mean'
        if 'distance_traveled' in df.columns:
            agg_dict['distance_traveled'] = 'sum'

        if not agg_dict:
            print("警告: 没有足够的列进行汇总统计")
            return pd.DataFrame()

        summary = df.groupby(['agent_name', 'agent_type', 'agent_series']).agg(agg_dict).round(4)
        summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col
                           for col in summary.columns.values]
        summary = summary.reset_index()

        return summary

    def save_processed_data(self, df, summary, output_dir='./processed_data'):
        """保存处理后的数据"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        full_path = output_path / 'all_monitors_combined.csv'
        df.to_csv(full_path, index=False)
        print(f"\n完整数据已保存至: {full_path}")

        if not summary.empty:
            summary_path = output_path / 'summary_statistics.csv'
            summary.to_csv(summary_path, index=False)
            print(f"汇总统计已保存至: {summary_path}")

        for agent_name in df['agent_name'].unique():
            agent_df = df[df['agent_name'] == agent_name]
            agent_path = output_path / f'{agent_name}_processed.csv'
            agent_df.to_csv(agent_path, index=False)
        print(f"各agent数据已保存至: {output_path}/")

        try:
            parquet_path = output_path / 'all_monitors_combined.parquet'
            df.to_parquet(parquet_path, index=False)
            print(f"Parquet格式已保存至: {parquet_path}")
        except:
            print("Parquet保存失败（可能需要安装pyarrow）")

        report_path = output_path / 'processing_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("数据处理报告\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"成功加载: {len(self.all_data)} 个文件\n")
            f.write(f"总行数: {len(df)}\n")
            f.write(f"总列数: {len(df.columns)}\n\n")

            if self.target_agents:
                f.write("指定处理的agent:\n")
                for agent in self.target_agents:
                    f.write(f"  - {agent}\n")
                f.write("\n")

            if self.failed_files:
                f.write(f"\n失败文件 ({len(self.failed_files)}):\n")
                for failed in self.failed_files:
                    f.write(f"  - {failed['agent']}\n")
                    for error in failed['errors']:
                        f.write(f"    {error}\n")

            if self.column_inconsistencies:
                f.write(f"\n列不一致警告 ({len(self.column_inconsistencies)}):\n")
                for inc in self.column_inconsistencies:
                    f.write(f"  - {inc['agent']}: {inc['inconsistencies']}\n")

        print(f"处理报告已保存至: {report_path}")

        return output_path


def main():
    """主函数"""

    # ============================================================
    # 配置区域 - 在这里修改您的设置
    # ============================================================

    # 1. 设置根目录（包含所有agent文件夹的目录）
    root_directory = r"D:\code\RL-frenet-trajectory-planning-in-CARLA-master\logs"

    # 2. 指定要处理的agent文件夹列表
    # 选项1: 只处理特定的agent（推荐）
    target_agents = [
        'agent_101_improved',
        'agent_102_improved',
        'agent_103_improved',
        'agent_201_improved',
        'agent_202_improved',
        'agent_203_improved',
        'agent_301_lyapunov',
        'agent_302_lyapunov',
        'agent_303_lyapunov',
    ]

    # 选项2: 处理所有agent（将上面的列表注释掉，取消下面的注释）
    # target_agents = None

    # ============================================================

    print("=" * 70)
    print("Monitor数据预处理工具")
    print("=" * 70)

    # 创建预处理器
    preprocessor = MonitorPreprocessor(root_directory, target_agents=target_agents)

    # 加载所有数据
    df = preprocessor.load_all_monitors()

    if df is None:
        print("\n❌ 处理失败：没有成功加载任何数据")
        return

    # 清洗数据
    df_cleaned = preprocessor.clean_data(df)

    # 生成汇总统计
    summary = preprocessor.generate_summary_stats(df_cleaned)

    # 打印基本信息
    print("\n" + "=" * 70)
    print("数据概览:")
    print("=" * 70)
    print(f"总行数: {len(df_cleaned)}")
    print(f"总列数: {len(df_cleaned.columns)}")
    print(f"Agent数量: {df_cleaned['agent_name'].nunique()}")
    print(f"\nAgent列表:")
    for agent in sorted(df_cleaned['agent_name'].unique()):
        count = len(df_cleaned[df_cleaned['agent_name'] == agent])
        agent_type = df_cleaned[df_cleaned['agent_name'] == agent]['agent_type'].iloc[0]
        print(f"  - {agent} ({agent_type}): {count} episodes")

    if not summary.empty:
        print("\n各Agent汇总统计:")
        display_cols = ['agent_name']
        if 'reward_mean' in summary.columns:
            display_cols.append('reward_mean')
        if 'success_mean' in summary.columns:
            display_cols.append('success_mean')
        if 'collision_sum' in summary.columns:
            display_cols.append('collision_sum')
        if 'training_step_max' in summary.columns:
            display_cols.append('training_step_max')

        print(summary[display_cols].to_string(index=False))

    # 保存数据
    output_dir = preprocessor.save_processed_data(df_cleaned, summary)

    print("\n" + "=" * 70)
    print("✓ 预处理完成！")
    print("=" * 70)
    print(f"\n数据已保存至: {output_dir}")
    print("\n可以使用以下代码加载处理后的数据:")
    print(">>> import pandas as pd")
    print(">>> df = pd.read_csv('processed_data/all_monitors_combined.csv')")
    print(">>> # 或使用parquet格式（更快）")
    print(">>> df = pd.read_parquet('processed_data/all_monitors_combined.parquet')")


if __name__ == "__main__":
    main()