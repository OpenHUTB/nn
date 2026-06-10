"""
统一的强化学习训练入口脚本
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def main():
    parser = argparse.ArgumentParser(
        description='强化学习训练框架',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 训练FrozenLake (Q-Learning)
    python train.py --env frozen_lake --algo q_learning --epochs 10000
    
    # 训练CartPole (Q-Learning)
    python train.py --env cartpole --algo q_learning --epochs 10000
    
    # 训练FrozenLake (PPO)
    python train.py --env frozen_lake --algo ppo --epochs 3000
    
    # 算法对比
    python train.py --benchmark --epochs 3000
        """
    )
    
    parser.add_argument('--env', type=str,
                        choices=['frozen_lake', 'cartpole', 'mountain_car', 'acrobot'],
                        default='frozen_lake',
                        help='训练环境')
    parser.add_argument('--algo', type=str,
                        choices=['q_learning', 'sarsa', 'dqn', 'reinforce', 'actor_critic', 'ppo'],
                        default='q_learning',
                        help='强化学习算法')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='训练轮数')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='折扣因子')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--test', action='store_true',
                        help='测试模式')
    parser.add_argument('--benchmark', action='store_true',
                        help='算法对比模式')
    parser.add_argument('--render', action='store_true',
                        help='渲染环境')
    
    args = parser.parse_args()
    
    if args.benchmark:
        run_benchmark(args)
    else:
        run_training(args)


def run_training(args):
    """运行训练"""
    if args.env == 'frozen_lake':
        from examples.frozen_lake_q_learning import train as train_fl
        from examples.frozen_lake_ppo import main as train_ppo
        
        if args.algo in ['q_learning', 'sarsa']:
            import subprocess
            cmd = [
                sys.executable, 
                'src/examples/frozen_lake_q_learning.py',
                '--epochs', str(args.epochs),
                '--alpha', str(args.alpha),
                '--gamma', str(args.gamma)
            ]
            if args.test:
                cmd.append('--test')
            subprocess.run(cmd)
        elif args.algo == 'ppo':
            import subprocess
            cmd = [
                sys.executable,
                'src/examples/frozen_lake_ppo.py',
                '--epochs', str(args.epochs),
                '--alpha', str(args.alpha),
                '--gamma', str(args.gamma)
            ]
            subprocess.run(cmd)
    
    elif args.env == 'cartpole':
        import subprocess
        cmd = [
            sys.executable,
            'src/examples/cartpole_q_learning.py',
            '--episodes', str(args.epochs),
            '--alpha', str(args.alpha),
            '--gamma', str(args.gamma)
        ]
        if args.test:
            cmd.append('--test')
        subprocess.run(cmd)


def run_benchmark(args):
    """运行算法对比"""
    import subprocess
    cmd = [
        sys.executable,
        'src/examples/benchmark_all.py',
        '--epochs', str(args.epochs)
    ]
    subprocess.run(cmd)


if __name__ == '__main__':
    main()
