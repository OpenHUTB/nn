import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='基于DQN的机器人自主导航系统')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='运行模式: train（训练）或 test（测试）')
    parser.add_argument('--episodes', type=int, default=500, help='训练轮数')
    parser.add_argument('--visualize', action='store_true', default=False, help='是否可视化')
    parser.add_argument('--no-visualize', dest='visualize', action='store_false', help='关闭可视化')

    args = parser.parse_args()

    os.makedirs('./results', exist_ok=True)

    if args.mode == 'train':
        from train import train
        train(episodes=args.episodes, visualize=args.visualize)
    elif args.mode == 'test':
        from test import test
        test(visualize=args.visualize)

if __name__ == '__main__':
    main()