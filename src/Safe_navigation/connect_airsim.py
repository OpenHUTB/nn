import airsim
import time


def main():
    print("=== 尝试连接 AirSimNH (使用 airsim 1.8.1) ===")
    print("重要：请先确保虚幻引擎中的 AirSimNH 已点击播放(Play)！\n")

    try:
        # 1. 创建客户端（1.8.1版本的唯一正确方式）
        client = airsim.CarClient()
        print("✓ 客户端对象创建成功")

        # 2. 确认连接（这会尝试与仿真器通信）
        client.confirmConnection()
        print("✓ 已连接到AirSim仿真服务器")

        # 3. 启用控制
        client.enableApiControl(True)
        print("✓ API控制已启用")

        # 4. 获取车辆状态，验证一切正常
        car_state = client.getCarState()
        print(f"✓ 车辆状态获取成功 - 速度: {car_state.speed} km/h")

        # 5. 【可选】简单控制演示
        print("\n>>> 连接成功！开始简单控制演示...")
        controls = airsim.CarControls()
        controls.throttle = 0.5
        client.setCarControls(controls)
        time.sleep(2)
        controls.brake = 1.0
        controls.throttle = 0.0
        client.setCarControls(controls)
        time.sleep(1)
        print("演示结束。")
        # 6. 释放控制
        client.enableApiControl(False)
        print("控制权已释放。")

    except ConnectionRefusedError:
        print("\n✗ 连接被拒绝。")
        print("  最可能的原因：虚幻引擎中的 AirSimNH 仿真没有启动。")
        print("  请打开虚幻引擎，加载AirSimNH项目，并点击顶部工具栏的蓝色【播放】(▶)按钮。")
    except Exception as e:
        print(f"\n✗ 连接过程中出错: {e}")
        print("  其他可能原因：防火墙阻止、端口占用或配置文件错误。")


if __name__ == "__main__":
    main()