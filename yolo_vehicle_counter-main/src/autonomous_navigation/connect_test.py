# airsim_connection.py
# 该模块用于连接 AirSim 模拟器（特别是 AbandonedPark 场景），并控制无人机执行基本任务。
# 包含连接、解锁、起飞、图像捕获、路径探索和降落等功能。

import airsim
import time


class AbandonedParkSimulator:
    """废弃公园场景的无人机模拟器控制类"""

    def __init__(self):
        """初始化：连接到本地 AirSim 模拟器并确认连接"""
        print("连接到AbandonedPark模拟器...")

        # 创建 AirSim 多旋翼客户端对象（默认连接本地）
        self.client = airsim.MultirotorClient()
        # 确认连接，如果失败会抛出异常
        self.client.confirmConnection()

        # 使用 ping 检查连接状态，正常应返回 True
        print(f"连接状态: {self.client.ping()}")
        print("模拟器已连接！")

    def ensure_drone_mode(self):
        """确保无人机处于受控模式（启用 API 控制并解锁）"""
        print("切换到无人机模式...")

        try:
            # 启用 API 控制，允许通过代码控制无人机
            self.client.enableApiControl(True)
            # 解锁无人机（模拟器中的电机解锁）
            self.client.armDisarm(True)
            print("无人机已解锁")
            return True
        except Exception as e:
            print(f"切换模式时出错: {e}")
            print("请确保模拟器中已选择无人机模式")
            return False

    def takeoff_and_hover(self, altitude=10):
        """起飞到指定高度并悬停"""
        print(f"起飞到 {altitude} 米高度...")

        # 起飞（异步操作，使用 join 等待完成）
        self.client.takeoffAsync().join()
        time.sleep(2)  # 等待稳定

        # 移动到指定高度（负值表示上升，因为 AirSim 中 Z 轴向下）
        self.client.moveToZAsync(-altitude, 3).join()
        time.sleep(1)  # 等待稳定

        print(f"已在 {altitude} 米高度悬停")

    def capture_park_image(self):
        """从无人机前置摄像头捕获图像并保存为文件"""
        print("捕获图像...")

        # 请求图像数据（前置摄像头 "0"，场景类型，不压缩，不使用像素格式转换）
        responses = self.client.simGetImages([
            airsim.ImageRequest(
                "0",                     # 相机名称，0 通常代表前置摄像头
                airsim.ImageType.Scene,  # 场景图像（彩色）
                False,                   # 不压缩为 JPEG
                False                    # 不进行像素格式转换
            )
        ])

        # 检查是否有图像返回
        if responses and len(responses) > 0:
            response = responses[0]

            # 将字节数据转换为 numpy 数组
            import numpy as np
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            # 根据图像高度和宽度重塑为三维数组 (H, W, 3)
            img_rgb = img1d.reshape(response.height, response.width, 3)

            # 保存图像（使用 OpenCV，BGR 格式，但这里 RGB 也可以保存）
            import cv2
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"park_capture_{timestamp}.jpg"
            cv2.imwrite(filename, img_rgb)
            print(f"图像已保存: {filename}")

            return img_rgb
        else:
            print("未能捕获图像")
            return None

    def explore_park(self):
        """执行一个简单的路径探索：依次飞往几个航点，并在每个航点拍照"""
        print("开始探索废弃公园...")

        # 定义航点列表 (x, y, z) ，z 为负值表示高度
        waypoints = [
            (20, 0, -10),   # 向前（x正方向）20米，保持高度
            (20, 15, -10),  # 向右（y正方向）15米
            (0, 15, -12),   # 向后20米，同时下降2米
            (0, 0, -10),    # 向左15米回到起点，恢复高度
        ]

        for x, y, z in waypoints:
            print(f"飞往位置: ({x}, {y}, {z})")
            # 以速度 3 m/s 飞往目标点
            self.client.moveToPositionAsync(x, y, z, 3).join()

            # 到达后捕获一张图像
            self.capture_park_image()
            time.sleep(1)   # 等待一下再飞向下一个点

        print("探索完成！")

    def cleanup(self):
        """清理资源：降落、锁定无人机、禁用 API 控制"""
        print("正在降落...")
        # 降落并等待完成
        self.client.landAsync().join()
        # 锁定无人机（上锁）
        self.client.armDisarm(False)
        # 禁用 API 控制，交还控制权给模拟器
        self.client.enableApiControl(False)
        print("无人机已降落")


# 快速测试脚本（当直接运行此文件时执行）
if __name__ == "__main__":
    print("=== AbandonedPark无人机测试 ===")

    # 1. 确保模拟器已经运行（用户需手动启动）
    input("请确保AbandonedPark.exe已运行，然后按回车继续...")

    # 2. 连接模拟器
    simulator = AbandonedParkSimulator()

    try:
        # 3. 切换到无人机模式（解锁）
        if simulator.ensure_drone_mode():
            # 4. 起飞至10米高度
            simulator.takeoff_and_hover(10)

            # 5. 捕获初始图像
            simulator.capture_park_image()

            # 6. 执行简单探索（飞航点并拍照）
            simulator.explore_park()

            # 7. 降落并清理
            simulator.cleanup()
    except KeyboardInterrupt:
        # 捕获 Ctrl+C，安全降落
        print("用户中断")
        simulator.cleanup()
    except Exception as e:
        # 其他异常处理
        print(f"发生错误: {e}")
        simulator.cleanup()