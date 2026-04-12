import argparse
import time

from agents.keyboard_controller import KeyboardController
from agents.random_walker import RandomWalker
from agents.vision_flyer import VisionFlyer
from client.drone_client import DroneClient


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--interval', default=1, type=int)
    args.add_argument('--move-type', default='velocity', type=str)
    args.add_argument('--save-path', default='./images', type=str)
    args.add_argument('--record', default=False, action='store_true', help='Enable recording')
    args.add_argument('--duration', default=30, type=int, help='Run duration in seconds')
    config = args.parse_args()

    client = DroneClient(config.interval, root_path=config.save_path)
    
    # 控制录制功能
    if config.record:
        print("录制功能已启用，开始记录飞行数据")
        # 尝试使用 startRecording 方法
        try:
            client.client.startRecording()
            print("录制已开始")
        except AttributeError:
            print("警告：当前 AirSim 版本可能不支持 startRecording 方法")
    else:
        print("录制功能已禁用，减少内存和磁盘使用")
    
    agent = KeyboardController(client, config.move_type)
    agent = RandomWalker(client, config.move_type, (-0.5, 0.5))  # 使用较小的速度范围
    # agent = VisionFlyer(client, config.move_type)
    
    # 运行指定时间
    print(f"开始运行，持续 {config.duration} 秒...")
    start_time = time.time()
    while time.time() - start_time < config.duration:
        agent.act()
        time.sleep(0.1)  # 小延迟，避免过于频繁的操作
    
    # 停止录制
    if config.record:
        try:
            client.client.stopRecording()
            print("录制已停止")
        except AttributeError:
            pass
    
    print("运行完成")
    client.destroy()