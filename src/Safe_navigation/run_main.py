import airsim
import time
import numpy as np
import cv2


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

        # 5. 获取并显示摄像头图像
        print("\n>>> 正在获取摄像头图像...")
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])

        if responses and responses[0].image_data_uint8:
            response = responses[0]
            # 将图像数据转换为numpy数组
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)

            # 显示图像
            cv2.imshow("AirSim Camera View", img_rgb)
            cv2.waitKey(1)
            print("✓ 摄像头图像已获取并显示")
        else:
            print("⚠ 无法获取摄像头图像")

        # 6. 精确90度转弯演示
        print("\n>>> 连接成功！开始精确90度转弯演示...")
        controls = airsim.CarControls()

        # 直行到路口
        controls.throttle = 0.5
        controls.steering = 0.0
        client.setCarControls(controls)
        print("直行前往路口...")

        # 在直行过程中持续显示摄像头图像
        for i in range(26):
            time.sleep(1)
            if i % 5 == 0:  # 每5秒更新一次摄像头图像
                responses = client.simGetImages([
                    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
                ])
                if responses and responses[0].image_data_uint8:
                    response = responses[0]
                    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                    img_rgb = img1d.reshape(response.height, response.width, 3)
                    cv2.imshow("AirSim Camera View", img_rgb)
                    cv2.waitKey(1)
                    print(f"  已行驶 {i + 1} 秒，摄像头图像已更新")

        # 到达路口，完全停车
        controls.throttle = 0.0
        controls.brake = 1.0
        client.setCarControls(controls)
        print("到达路口，停车...")
        time.sleep(1)

        # 缓慢起步并适度转向
        controls.brake = 0.0
        controls.throttle = 0.25
        controls.steering = 0.7
        client.setCarControls(controls)
        print("缓慢起步转弯...")
        time.sleep(4)

        # 稍微回正一点方向盘
        controls.steering = 0.5
        client.setCarControls(controls)
        print("调整转向角度...")
        time.sleep(2)

        # 完全回正方向盘
        controls.steering = 0.0
        client.setCarControls(controls)
        print("转弯完成，直行...")
        time.sleep(5)

        # 缓慢减速停止
        controls.throttle = 0.2
        client.setCarControls(controls)
        print("减速中...")
        time.sleep(2)

        controls.brake = 1.0
        controls.throttle = 0.0
        client.setCarControls(controls)
        print("停车...")
        time.sleep(1)

        # 关闭图像显示窗口
        cv2.destroyAllWindows()
        print("演示结束。")

        # 7. 释放控制
        client.enableApiControl(False)
        print("控制权已释放。")

    except ConnectionRefusedError:
        print("\n✗ 连接被拒绝。")
        print("  最可能的原因：虚幻引擎中的 AirSimNH 仿真没有启动。")
        print("  请打开虚幻引擎，加载AirSimNH项目，并点击顶部工具栏的蓝色【播放】(▶)按钮。")
    except Exception as e:
        print(f"\n✗ 连接过程中出错: {e}")
        print("  其他可能原因：防火墙阻止、端口占用或配置文件错误。")
    finally:
        # 确保窗口被关闭
        try:
            cv2.destroyAllWindows()
        except:
            pass


if __name__ == "__main__":
    main()