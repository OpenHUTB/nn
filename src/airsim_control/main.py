import argparse
import time

from agents.keyboard_controller import KeyboardController
from agents.dqn_agent import DQNAgent
from client.drone_client import DroneClient


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--mode', default='keyboard',
                      choices=['keyboard', 'dqn-train', 'dqn-test'],
                      help='运行模式')
    args.add_argument('--interval', default=1, type=int)
    args.add_argument('--move-type', default='velocity', type=str)
    args.add_argument('--save-path', default='./images', type=str)
    args.add_argument('--record', default=False, action='store_true', help='Enable recording')
    args.add_argument('--duration', default=30, type=int, help='Run duration in seconds')
    args.add_argument('--episodes', default=100, type=int, help='DQN training episodes')
    args.add_argument('--model-path', default='./models/dqn_model.pth', help='DQN model path')
    config = args.parse_args()

    client = DroneClient(config.interval, root_path=config.save_path)

    if config.mode == 'keyboard':
        # 键盘控制模式
        if config.record:
            print("录制功能已启用，开始记录飞行数据")
            try:
                client.client.startRecording()
                print("录制已开始")
            except AttributeError:
                print("警告：当前 AirSim 版本可能不支持 startRecording 方法")
        else:
            print("录制功能已禁用，减少内存和磁盘使用")

        agent = KeyboardController(client, config.move_type)

        print(f"开始运行，持续 {config.duration} 秒...")
        start_time = time.time()
        while time.time() - start_time < config.duration:
            agent.act()
            time.sleep(0.1)

        if config.record:
            try:
                client.client.stopRecording()
                print("录制已停止")
            except AttributeError:
                pass

        print("运行完成")

    elif config.mode == 'dqn-train':
        # DQN 训练模式
        print("DQN 训练模式")
        agent = DQNAgent(client, config.move_type)
        agent.train(episodes=config.episodes, save_path='./models')

    elif config.mode == 'dqn-test':
        # DQN 测试模式
        print("DQN 测试模式")
        agent = DQNAgent(client, config.move_type)
        try:
            agent.load(config.model_path)
        except FileNotFoundError:
            print(f"模型文件不存在: {config.model_path}")
            print("请先使用 --mode dqn-train 进行训练")
            client.destroy()
            exit(1)
        agent.run(episodes=10)

    client.destroy()