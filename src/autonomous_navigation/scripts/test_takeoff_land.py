# scripts/test_takeoff_land.py
"""
测试无人机的起飞、悬停和降落功能。
运行前请确保 AirSim 模拟器已启动。
"""
import airsim
import time
import sys


def main():
    print("=" * 50)
    print("起飞/降落测试")
    print("=" * 50)
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("✓ 已连接模拟器")

    # 解锁
    client.enableApiControl(True)
    client.armDisarm(True)
    print("✓ 无人机已解锁")

    # 起飞
    print("起飞至 5 米...")
    client.takeoffAsync().join()
    time.sleep(2)
    client.moveToZAsync(-5, 2).join()
    print("✓ 已到达 5 米高度，悬停 3 秒")
    time.sleep(3)

    # 降落
    print("降落...")
    client.landAsync().join()
    print("✓ 降落完成")

    # 锁定
    client.armDisarm(False)
    client.enableApiControl(False)
    print("测试结束")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"错误: {e}")