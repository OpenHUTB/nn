# airsim_connection.py
import airsim
import time


class AbandonedParkSimulator:
    def __init__(self):
        print("连接到AbandonedPark模拟器...")

        # 连接到本地的AbandonedPark模拟器
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        # 检查连接状态
        print(f"连接状态: {self.client.ping()}")

        print("模拟器已连接！")

    def ensure_drone_mode(self):
        """确保切换到无人机模式"""
        print("切换到无人机模式...")

        # 尝试解锁无人机
        try:
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            print("无人机已解锁")
            return True
        except Exception as e:
            print(f"切换模式时出错: {e}")
            print("请确保模拟器中已选择无人机模式")
            return False

    def takeoff_and_hover(self, altitude=10):
        """起飞并悬停"""
        print(f"起飞到 {altitude} 米高度...")

        # 起飞
        self.client.takeoffAsync().join()
        time.sleep(2)

        # 移动到指定高度
        self.client.moveToZAsync(-altitude, 3).join()
        time.sleep(1)

        print(f"已在 {altitude} 米高度悬停")

    def capture_park_image(self):
        """捕获废弃公园图像"""
        print("捕获图像...")

        # 从相机获取图像
        responses = self.client.simGetImages([
            airsim.ImageRequest(
                "0",  # 前置摄像头
                airsim.ImageType.Scene,
                False, False  # 不压缩
            )
        ])

        if responses and len(responses) > 0:
            response = responses[0]

            # 转换为numpy数组
            import numpy as np
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)

            # 保存图像
            import cv2
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"park_capture_{timestamp}.jpg", img_rgb)
            print(f"图像已保存: park_capture_{timestamp}.jpg")

            return img_rgb
        else:
            print("未能捕获图像")
            return None

    def explore_park(self):
        """探索公园的简单路径"""
        print("开始探索废弃公园...")

        # 定义探索路径（围绕公园）
        waypoints = [
            (20, 0, -10),  # 向前20米
            (20, 15, -10),  # 向右15米
            (0, 15, -12),  # 向后20米，下降2米
            (0, 0, -10),  # 向左15米，回到起点
        ]

        for x, y, z in waypoints:
            print(f"飞往位置: ({x}, {y}, {z})")
            self.client.moveToPositionAsync(x, y, z, 3).join()

            # 在每个位置捕获图像
            self.capture_park_image()
            time.sleep(1)

        print("探索完成！")

    def cleanup(self):
        """清理资源"""
        print("正在降落...")
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        print("无人机已降落")


# 快速测试脚本
if __name__ == "__main__":
    print("=== AbandonedPark无人机测试 ===")

    # 1. 确保模拟器已经运行
    input("请确保AbandonedPark.exe已运行，然后按回车继续...")

    # 2. 连接模拟器
    simulator = AbandonedParkSimulator()

    try:
        # 3. 切换到无人机模式
        if simulator.ensure_drone_mode():
            # 4. 起飞
            simulator.takeoff_and_hover(10)

            # 5. 捕获初始图像
            simulator.capture_park_image()

            # 6. 简单探索
            simulator.explore_park()

            # 7. 降落
            simulator.cleanup()
    except KeyboardInterrupt:
        print("用户中断")
        simulator.cleanup()
    except Exception as e:
        print(f"发生错误: {e}")
        simulator.cleanup()