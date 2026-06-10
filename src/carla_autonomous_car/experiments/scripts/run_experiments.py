"""
run_experiments.py - 完整的实验运行系统（Python版本）

使用方法：
    # 快速测试（每个方法100步）
    python experiments/scripts/run_experiments.py --mode quick_test

    # 运行所有实验（每个方法10000步）
    python experiments/scripts/run_experiments.py --mode all

    # 运行单个实验
    python experiments/scripts/run_experiments.py --mode single --method lyapunov --seed 1
"""

import os
import sys
import time
import socket
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import json


class ExperimentRunner:
    """实验运行器"""

    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = Path.cwd()
        else:
            base_dir = Path(base_dir)

        self.base_dir = base_dir
        self.results_dir = base_dir / 'experiments' / 'results'

        # 实验配置
        self.methods = {
            'baseline': {
                'cfg_file': 'tools/cfgs/experiment_baseline.yaml',
                'reward_system': 'improved',
                'agent_id_base': 100,
            },
            'improved': {
                'cfg_file': 'tools/cfgs/experiment_improved.yaml',
                'reward_system': 'improved',
                'agent_id_base': 200,
            },
            'lyapunov': {
                'cfg_file': 'tools/cfgs/experiment_lyapunov.yaml',
                'reward_system': 'lyapunov',
                'agent_id_base': 300,
            }
        }

        # CARLA配置
        self.carla_host = '127.0.0.1'
        self.carla_port = 2000
        self.tm_port = 8000

        # ⭐ 修改：实验参数
        self.num_timesteps = 10000  # 默认训练步数：10000步
        self.quick_test_timesteps = 100  # 快速测试步数：100步
        self.num_seeds = 3

        # 日志
        self.log_file = None

    def check_carla_server(self):
        """检查CARLA服务器是否运行"""
        print("🔍 Checking CARLA server...")

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((self.carla_host, self.carla_port))
        sock.close()

        if result == 0:
            print(f"✅ CARLA server is running on {self.carla_host}:{self.carla_port}")
            return True
        else:
            print(f"❌ CARLA server not running on {self.carla_host}:{self.carla_port}")
            print("\nPlease start CARLA server first:")
            print("  Windows: CarlaUE4.exe -windowed -ResX=800 -ResY=600")
            print("  Linux:   ./CarlaUE4.sh -windowed -ResX=800 -ResY=600")
            return False

    def create_directories(self):
        """创建必要的目录结构"""
        print("📁 Creating directory structure...")

        for method in self.methods.keys():
            method_dir = self.results_dir / method
            (method_dir / 'logs').mkdir(parents=True, exist_ok=True)
            (method_dir / 'models').mkdir(parents=True, exist_ok=True)
            (method_dir / 'plots').mkdir(parents=True, exist_ok=True)

        # 创建分析和图表目录
        (self.base_dir / 'experiments' / 'analysis' / 'data').mkdir(parents=True, exist_ok=True)
        (self.base_dir / 'experiments' / 'figures' / 'learning_curves').mkdir(parents=True, exist_ok=True)
        (self.base_dir / 'experiments' / 'figures' / 'comparisons').mkdir(parents=True, exist_ok=True)

        print("✅ Directories created")

    def run_single_experiment(self, method, seed, num_timesteps=None):
        """
        运行单个实验

        Args:
            method: 方法名称 ('baseline', 'improved', 'lyapunov')
            seed: 随机种子
            num_timesteps: 训练步数（None使用默认值）
        """
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}. Choose from {list(self.methods.keys())}")

        method_config = self.methods[method]
        agent_id = method_config['agent_id_base'] + seed

        if num_timesteps is None:
            num_timesteps = self.num_timesteps

        # 构建命令
        cmd = [
            sys.executable,  # 使用当前Python解释器
            'run_lyapunov.py',
            '--cfg_file', method_config['cfg_file'],
            '--agent_id', str(agent_id),
            '--reward_system', method_config['reward_system'],
            '--num_timesteps', str(num_timesteps),
            '--carla_host', self.carla_host,
            '--carla_port', str(self.carla_port),
            '--tm_port', str(self.tm_port),
            '--verbosity', '1',
        ]

        # 日志文件
        log_file = self.results_dir / method / 'logs' / f'training_seed{seed}.log'

        # 打印信息
        print("\n" + "=" * 80)
        print(f"🔹 Running {method.upper()} - Seed {seed} (Agent {agent_id})")
        print("=" * 80)
        print(f"Command: {' '.join(cmd)}")
        print(f"Log file: {log_file}")
        print(f"Timesteps: {num_timesteps:,}")
        print("-" * 80)

        # 运行实验
        start_time = time.time()

        with open(log_file, 'w', encoding='utf-8') as f:  # ⭐ 添加encoding='utf-8'
            # 写入开始时间
            f.write(f"# Experiment started at {datetime.now()}\n")
            f.write(f"# Command: {' '.join(cmd)}\n")
            f.write("#" + "=" * 78 + "\n\n")
            f.flush()

            try:
                # 运行训练，实时输出到终端和文件
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    encoding='utf-8',  # ⭐ 添加encoding
                    errors='replace'   # ⭐ 添加错误处理
                )

                # 实时读取输出
                for line in process.stdout:
                    print(line, end='')  # 输出到终端
                    f.write(line)  # 写入日志文件
                    f.flush()

                # 等待进程结束
                process.wait()
                return_code = process.returncode

            except KeyboardInterrupt:
                print("\n⚠️  Interrupted by user")
                process.terminate()
                return_code = -1
            except Exception as e:
                print(f"\n❌ Error: {e}")
                return_code = -1

        # 运行时间
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        # 打印结果
        print("\n" + "-" * 80)
        if return_code == 0:
            print(f"✅ {method.upper()} Seed {seed} completed successfully")
        else:
            print(f"❌ {method.upper()} Seed {seed} failed (return code: {return_code})")
        print(f"⏱️  Time elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print("=" * 80 + "\n")

        # 等待CARLA重置
        print("⏳ Waiting 5 seconds for CARLA reset...")
        time.sleep(5)

        return return_code == 0

    def run_all_experiments(self, num_timesteps=None):
        """运行所有实验"""
        print("\n" + "=" * 80)
        print("🚀 STARTING COMPLETE EXPERIMENT PIPELINE")
        print("=" * 80)
        print(f"Methods: {list(self.methods.keys())}")
        print(f"Seeds: {self.num_seeds}")
        print(f"Timesteps per experiment: {num_timesteps or self.num_timesteps:,}")
        print("=" * 80 + "\n")

        # 检查CARLA
        if not self.check_carla_server():
            return False

        # 创建目录
        self.create_directories()

        # 记录实验开始时间
        start_time = time.time()
        experiment_log = {
            'start_time': datetime.now().isoformat(),
            'methods': list(self.methods.keys()),
            'num_seeds': self.num_seeds,
            'num_timesteps': num_timesteps or self.num_timesteps,
            'results': {}
        }

        # 运行所有实验
        total_experiments = len(self.methods) * self.num_seeds
        completed_experiments = 0

        for method in self.methods.keys():
            print("\n" + "🔷" * 40)
            print(f"EXPERIMENT: {method.upper()}")
            print("🔷" * 40 + "\n")

            method_results = []

            for seed in range(1, self.num_seeds + 1):
                success = self.run_single_experiment(method, seed, num_timesteps)
                method_results.append({
                    'seed': seed,
                    'success': success
                })

                completed_experiments += 1
                print(f"\n📊 Progress: {completed_experiments}/{total_experiments} experiments completed\n")

            experiment_log['results'][method] = method_results

        # 计算总时间
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)

        experiment_log['end_time'] = datetime.now().isoformat()
        experiment_log['total_time_seconds'] = total_time

        # 保存实验日志
        log_path = self.results_dir.parent / 'experiment_log.json'
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_log, f, indent=2, ensure_ascii=False)

        # 整理结果
        self.organize_results()

        # 打印总结
        print("\n" + "=" * 80)
        print("🎉 ALL EXPERIMENTS COMPLETED!")
        print("=" * 80)
        print(f"⏱️  Total time: {hours:02d}:{minutes:02d}")
        print(f"📁 Results saved to: {self.results_dir}")
        print(f"📋 Experiment log: {log_path}")
        print("\nNext steps:")
        print("  1. python experiments/scripts/analyze_results.py")
        print("  2. python experiments/scripts/plot_results.py")
        print("  3. python experiments/scripts/generate_report.py")
        print("=" * 80 + "\n")

        return True

    def organize_results(self):
        """整理实验结果（复制monitor.csv等）"""
        print("\n" + "=" * 80)
        print("📋 ORGANIZING RESULTS")
        print("=" * 80)

        logs_dir = self.base_dir / 'logs'

        for method, config in self.methods.items():
            print(f"\n🔹 Organizing {method} results...")

            for seed in range(1, self.num_seeds + 1):
                agent_id = config['agent_id_base'] + seed
                reward_system = config['reward_system']

                # 源文件路径
                source_dir = logs_dir / f'agent_{agent_id}_{reward_system}'
                source_monitor = source_dir / 'monitor.csv'

                # 目标文件路径
                dest_monitor = self.results_dir / method / 'logs' / f'monitor_seed{seed}.csv'

                # 复制monitor.csv
                if source_monitor.exists():
                    import shutil
                    shutil.copy2(source_monitor, dest_monitor)
                    print(f"  ✓ Copied monitor.csv for seed {seed}")
                else:
                    print(f"  ✗ Warning: {source_monitor} not found")

        print("\n✅ Results organized!")

    def quick_test(self):
        """⭐ 修改：快速测试（每个方法100步）"""
        print("\n" + "=" * 80)
        print("🧪 QUICK TEST MODE")
        print("=" * 80)
        print(f"Running baseline and lyapunov with {self.quick_test_timesteps} timesteps each")
        print("=" * 80 + "\n")

        if not self.check_carla_server():
            return False

        self.create_directories()

        # 测试baseline
        print("\n🔹 Testing BASELINE...")
        success_baseline = self.run_single_experiment('baseline', 1, num_timesteps=self.quick_test_timesteps)

        # 测试lyapunov
        print("\n🔹 Testing LYAPUNOV...")
        success_lyapunov = self.run_single_experiment('lyapunov', 1, num_timesteps=self.quick_test_timesteps)

        # 总结
        print("\n" + "=" * 80)
        if success_baseline and success_lyapunov:
            print("✅ QUICK TEST PASSED!")
        else:
            print("❌ QUICK TEST FAILED!")
        print("=" * 80)

        print("\nCheck logs at:")
        print(f"  - {self.results_dir / 'baseline' / 'logs' / 'training_seed1.log'}")
        print(f"  - {self.results_dir / 'lyapunov' / 'logs' / 'training_seed1.log'}")
        print("\nIf successful, run full experiments with:")
        print("  python experiments/scripts/run_experiments.py --mode all")
        print("=" * 80 + "\n")

        return success_baseline and success_lyapunov


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Run Lyapunov experiments')

    parser.add_argument('--mode', type=str, default='quick_test',
                        choices=['quick_test', 'all', 'single'],
                        help='Execution mode')

    parser.add_argument('--method', type=str, default='lyapunov',
                        choices=['baseline', 'improved', 'lyapunov'],
                        help='Method to run (for single mode)')

    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (for single mode)')

    parser.add_argument('--num_timesteps', type=int, default=None,
                        help='Number of timesteps (default: 10000 for all, 100 for quick_test)')

    parser.add_argument('--num_seeds', type=int, default=3,
                        help='Number of random seeds (for all mode)')

    parser.add_argument('--carla_host', type=str, default='127.0.0.1',
                        help='CARLA host')

    parser.add_argument('--carla_port', type=int, default=2000,
                        help='CARLA port')

    args = parser.parse_args()

    # 创建运行器
    runner = ExperimentRunner()
    runner.carla_host = args.carla_host
    runner.carla_port = args.carla_port
    runner.num_seeds = args.num_seeds

    # 根据模式运行
    if args.mode == 'quick_test':
        runner.quick_test()

    elif args.mode == 'all':
        # ⭐ 修改：如果未指定timesteps，使用默认的10000
        num_timesteps = args.num_timesteps if args.num_timesteps is not None else runner.num_timesteps
        runner.run_all_experiments(num_timesteps=num_timesteps)

    elif args.mode == 'single':
        # ⭐ 修改：单个实验默认也是10000步
        num_timesteps = args.num_timesteps if args.num_timesteps is not None else runner.num_timesteps
        runner.run_single_experiment(args.method, args.seed, num_timesteps)

    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == '__main__':
    main()