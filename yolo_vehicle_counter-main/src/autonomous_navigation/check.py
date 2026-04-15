# check_status.py
"""
无人机状态检查脚本
功能：连接 AirSim 模拟器，获取并显示无人机当前状态，然后解锁无人机，准备后续任务。
适用于 AbandonedPark 环境。
"""

import airsim
import time
import sys


def check_connection():
    """
    连接到 AirSim 模拟器，并确认连接成功。
    如果连接失败，打印错误信息并退出程序。
    """
    print("=" * 50)
    print("无人机状态检查")
    print("=" * 50)

    try:
        # 创建多旋翼客户端并确认连接
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("✓ 已连接到 AbandonedPark 模拟器")
        return client
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        print("请确保模拟器已启动并处于无人机模式。")
        sys.exit(1)


def print_drone_state(client):
    """
    获取并打印无人机的当前状态，包括位置、速度、电池电量和碰撞信息。
    """
    try:
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        print(f"当前位置: X={pos.x_val:.1f}, Y={pos.y_val:.1f}, Z={pos.z_val:.1f}")
        print(f"当前速度: {state.speed:.2f} m/s")
        print(f"电池电量: {state.battery:.1f}%")
        print(f"是否碰撞: {state.collision.has_collided}")
    except Exception as e:
        print(f"获取状态失败: {e}")


def unlock_drone(client):
    """
    解锁无人机：启用 API 控制并解除电机的锁定。
    这是执行任何飞行动作前的必要步骤。
    """
    print("\n解锁无人机...")
    try:
        client.enableApiControl(True)   # 允许通过 API 控制无人机
        client.armDisarm(True)          # 解锁电机（模拟器中的上锁/解锁）
        print("✓ 无人机已解锁，准备就绪")
    except Exception as e:
        print(f"✗ 解锁失败: {e}")
        # 尝试释放控制，避免影响模拟器
        client.enableApiControl(False)
        sys.exit(1)


def main():
    """
    主函数：连接模拟器、显示状态、解锁无人机。
    """
    # 1. 连接模拟器
    client = check_connection()

    # 2. 获取并显示当前状态
    print_drone_state(client)

    # 3. 解锁无人机
    unlock_drone(client)

    # 4. 完成提示
    print("\n" + "=" * 50)
    print("状态检查完成！无人机可以正常控制")
    print("=" * 50)


if __name__ == "__main__":
    main()