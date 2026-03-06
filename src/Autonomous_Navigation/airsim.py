# airsim_connection.py
"""
无人机与 AirSim 模拟器（AbandonedPark 场景）的连接与控制模块。
提供连接、解锁、起飞、图像捕获、路径探索和降落等功能。
"""

import airsim
import time
import os
import sys
from datetime import datetime
import numpy as np
import cv2


class AbandonedParkSimulator:
    """废弃公园场景的无人机模拟器控制类"""

    def __init__(self, auto_connect=True):
        """
        初始化：可选择自动连接模拟器。
        :param auto_connect: 是否在初始化时自动连接
        """
        self.client = None
        self.connected = False
        self.flying = False

        if auto_connect:
            self.connect()

    def connect(self, max_retries=3, retry_delay=1):
        """
        连接到本地 AirSim 模拟器，支持重试机制。
        :param max_retries: 最大重试次数
        :param retry_delay: 重试间隔（秒）
        :return: 是否连接成功
        """
        print("正在连接到 AbandonedPark 模拟器...")
        for attempt in range(1, max_retries + 1):
            try:
                self.client = airsim.MultirotorClient()
                self.client.confirmConnection()
                # 使用 ping 验证连接
                if self.client.ping():
                    self.connected = True
                    print(f"✓ 连接成功（尝试 {attempt}）")
                    print(f"连接状态: {self.client.ping()}")
                    return True
            except Exception as e:
                print(f"✗ 连接尝试 {attempt} 失败: {e}")
                if attempt < max_retries:
                    print(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
        self.connected = False
        print("✗ 无法连接到模拟器，请检查模拟器是否已启动。")
        return False

    def ensure_drone_mode(self):
        """确保无人机处于受控模式（启用 API 控制并解锁）"""
        if not self.connected or self.client is None:
            print("错误：未连接到模拟器")
            return False

        print("切换到无人机模式...")
        try:
            # 启用 API 控制
            self.client.enableApiControl(True)
            # 解锁电机
            self.client.armDisarm(True)
            print("✓ 无人机已解锁")
            return True
        except Exception as e:
            print(f"✗ 切换模式时出错: {e}")
            print("提示：请确保模拟器中已选择无人机模式（按 F2 切换）")
            return False

    def takeoff_and_hover(self, altitude=10, timeout=10):
        """
        起飞到指定高度并悬停。
        :param altitude: 目标高度（米）
        :param timeout: 起飞超时时间（秒）
        """
        if not self.connected:
            print("错误：未连接到模拟器")
            return

        print(f"起飞到 {altitude} 米高度...")
        try:
            # 起飞
            self.client.takeoffAsync().join(timeout=timeout)
            time.sleep(2)  # 等待稳定

            # 移动到指定高度（AirSim 中 Z 轴向下，负值表示上升）
            self.client.moveToZAsync(-altitude, 3).join(timeout=timeout)
            time.sleep(1)

            self.flying = True
            print(f"✓ 已在 {altitude} 米高度悬停")
        except Exception as e:
            print(f"✗ 起飞失败: {e}")
            self.flying = False

    def capture_image(self, camera_name="0", save_dir="captures"):
        """
        从指定摄像头捕获图像并保存为文件。
        :param camera_name: 摄像头名称，默认为前置摄像头 "0"
        :param save_dir: 保存图像的目录
        :return: 图像 numpy 数组，如果失败则返回 None
        """
        if not self.connected:
            print("错误：未连接到模拟器")
            return None

        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)

        try:
            # 请求图像
            responses = self.client.simGetImages([
                airsim.ImageRequest(
                    camera_name,
                    airsim.ImageType.Scene,
                    False,  # 不压缩为 JPEG
                    False   # 不进行像素格式转换
                )
            ])

            if not responses or len(responses) == 0:
                print("✗ 未收到图像响应")
                return None

            response = responses[0]
            # 将字节数据转换为 numpy 数组
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            # 根据图像尺寸重塑
            img_rgb = img1d.reshape(response.height, response.width, 3)

            # 生成文件名（使用时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = os.path.join(save_dir, f"capture_{timestamp}.jpg")
            # 保存图像（OpenCV 使用 BGR，但 RGB 也可保存）
            cv2.imwrite(filename, img_rgb)
            print(f"✓ 图像已保存: {filename}")

            return img_rgb
        except Exception as e:
            print(f"✗ 图像捕获失败: {e}")
            return None

    def explore_park(self, waypoints=None):
        """
        执行路径探索：依次飞往航点并在每个航点拍照。
        :param waypoints: 航点列表，每个元素为 (x, y, z) 元组，z 为负值
                          若为 None，则使用默认路径
        """
        if not self.flying:
            print("错误：无人机尚未起飞")
            return

        if waypoints is None:
            # 默认探索路径（围绕公园）
            waypoints = [
                (20, 0, -10),   # 向前（x正方向）20米，保持高度
                (20, 15, -10),  # 向右（y正方向）15米
                (0, 15, -12),   # 向后20米，同时下降2米
                (0, 0, -10),    # 向左15米回到起点，恢复高度
            ]

        print("开始探索废弃公园...")
        for idx, (x, y, z) in enumerate(waypoints, 1):
            print(f"[{idx}/{len(waypoints)}] 飞往位置: ({x}, {y}, {z})")
            try:
                # 以速度 3 m/s 飞往目标点
                self.client.moveToPositionAsync(x, y, z, 3).join()
                # 到达后拍照
                self.capture_image()
                time.sleep(1)  # 短暂停留
            except Exception as e:
                print(f"✗ 移动至航点时出错: {e}")
                break
        print("探索完成！")

    def land(self):
        """降落无人机并清理资源"""
        if not self.connected or self.client is None:
            return

        print("正在降落...")
        try:
            if self.flying:
                self.client.landAsync().join()
                self.flying = False
            # 锁定无人机
            self.client.armDisarm(False)
            # 禁用 API 控制
            self.client.enableApiControl(False)
            print("✓ 无人机已降落并锁定")
        except Exception as e:
            print(f"✗ 降落过程中出错: {e}")

    def cleanup(self):
        """清理资源（同 land，作为别名）"""
        self.land()


# 快速测试脚本（当直接运行此文件时执行）
if __name__ == "__main__":
    print("=== AbandonedPark 无人机测试 ===")
    print("请确保 AbandonedPark.exe 已运行并按 F2 切换到无人机模式。")
    input("按回车键继续...")

    # 创建模拟器对象并自动连接
    sim = AbandonedParkSimulator(auto_connect=True)

    if not sim.connected:
        print("无法连接，退出测试。")
        sys.exit(1)

    try:
        # 切换无人机模式（解锁）
        if not sim.ensure_drone_mode():
            sys.exit(1)

        # 起飞至 10 米高度
        sim.takeoff_and_hover(10)

        # 捕获一张初始图像
        sim.capture_image()

        # 执行默认路径探索
        sim.explore_park()

    except KeyboardInterrupt:
        print("\n用户中断，正在降落...")
    except Exception as e:
        print(f"\n发生未预期的错误: {e}")
    finally:
        # 确保降落清理
        sim.cleanup()
        print("测试结束。")