#!/usr/bin/env python
import carla
import time
import os

# 获取当前脚本所在目录（文件一定保存在这里）
current_dir = os.path.dirname(os.path.abspath(__file__))
rec_file = os.path.join(current_dir, "test1.rec")

def main():
    try:
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(5.0)

        print("✅ 连接 CARLA 成功")
        print(f"▶️  开始录制，文件保存在：{rec_file}")

        client.start_recorder(rec_file)
        time.sleep(15)
        client.stop_recorder()

        print("✅ 录制完成！")
        print(f"📁 文件位置：{rec_file}")

    except Exception as e:
        print(f"❌ 错误：{e}")

if __name__ == '__main__':
    main()