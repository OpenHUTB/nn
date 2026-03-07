#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
airsim_connection.py - AbandonedPark 模拟器无人机连接与控制模块

提供与 AirSim 模拟器（AbandonedPark 场景）的连接、解锁、起飞、
图像捕获、路径探索和降落等基础功能。
"""

import airsim
import time
import os
import sys
from typing import Optional, Tuple, List

# 常量定义
DEFAULT_ALTITUDE = 10          # 默认起飞高度（米）
MOVE_SPEED = 3.0                # 默认移动速度 (m/s)
STABILIZE_DELAY = 2.0            # 起飞后稳定等待时间（秒）
POST_MOVE_DELAY = 1.0            # 移动后等待时间（秒）
IMAGE_SAVE_DIR = "captures"      # 默认图像保存目录


class AbandonedParkSimulator:
    """废弃公园场景的无人机模拟器控制类"""

    def __init__(self, ip: str = "127.0.0.1", port: int = 41451):
        """
        初始化控制器，可选择自动连接

        :param ip: 模拟器 IP 地址，默认本地
        :param port: 模拟器端口，默认 41451
        """
        self.ip = ip
        self.port = port
        self.client: Optional[airsim.MultirotorClient] = None
        self.is_connected = False
        self.is_armed = False        # 是否已解锁
        self.is_flying = False        # 是否在飞行

        print(f"初始化 AbandonedPark 控制器，目标 {ip}:{port}")
        self._connect()

    # ------------------------------------------------------------------
    # 私有连接方法
    # ------------------------------------------------------------------
    def _connect(self, timeout: float = 10.0) -> bool:
        """
        连接到模拟器（内部调用）

        :param timeout: 连接超时时间（秒）
        :return: 是否连接成功
        """
        try:
            self.client = airsim.MultirotorClient(ip=self.ip, port=self.port)
            self.client.confirmConnection()
            # 使用 ping 测试连接
            ping_time = self.client.ping()
            print(f"✓ 连接成功！响应时间: {ping_time} ms")
            self.is_connected = True
            return True
        except Exception as e:
            print(f"✗ 连接失败: {e}")
            self.is_connected = False
            return False

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def ensure_drone_mode(self) -> bool:
        """
        确保无人机处于受控模式（启用 API 控制并解锁）

        :return: 是否成功切换
        """
        if not self.is_connected:
            print("错误：未连接到模拟器")
            return False

        print("尝试切换到无人机模式...")
        try:
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            self.is_armed = True
            print("✓ 无人机已解锁")
            return True
        except Exception as e:
            print(f"✗ 切换失败: {e}")
            print("提示：请确保模拟器中已选择无人机模式（按 F2 切换）")
            return False

    def takeoff_and_hover(self, altitude: float = DEFAULT_ALTITUDE) -> bool:
        """
        起飞到指定高度并悬停

        :param altitude: 目标高度（米）
        :return: 是否成功
        """
        if not self.is_connected:
            print("错误：未连接到模拟器")
            return False
        if not self.is_armed:
            print("错误：无人机未解锁，请先调用 ensure_drone_mode()")
            return False

        print(f"起飞到 {altitude} 米高度...")
        try:
            # 起飞
            self.client.takeoffAsync().join()
            time.sleep(STABILIZE_DELAY)

            # 上升到指定高度（AirSim 中 Z 轴向下，负值表示上升）
            self.client.moveToZAsync(-altitude, MOVE_SPEED).join()
            time.sleep(POST_MOVE_DELAY)

            self.is_flying = True
            print(f"✓ 已在 {altitude} 米高度悬停")
            return True
        except Exception as e:
            print(f"✗ 起飞失败: {e}")
            return False

    def capture_park_image(self, save_dir: str = IMAGE_SAVE_DIR) -> Optional[airsim.ImageRequest]:
        """
        从无人机前置摄像头捕获图像并保存为文件

        :param save_dir: 保存图像的目录（自动创建）
        :return: 图像对象（numpy 数组），失败返回 None
        """
        if not self.is_connected:
            print("错误：未连接到模拟器")
            return None

        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)

        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])
            if not responses:
                print("✗ 未收到图像响应")
                return None

            response = responses[0]
            # 转换为 numpy 数组
            import numpy as np
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)

            # 保存图像（使用 OpenCV）
            import cv2
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"park_capture_{timestamp}.jpg")
            cv2.imwrite(filename, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            print(f"✓ 图像已保存: {filename}")

            return img_rgb
        except Exception as e:
            print(f"✗ 图像捕获失败: {e}")
            return None

    def explore_park(self, waypoints: Optional[List[Tuple[float, float, float]]] = None) -> bool:
        """
        执行路径探索：依次飞往航点并在每个航点拍照

        :param waypoints: 航点列表，每个元素为 (x, y, z) 元组（z 为负值表示高度）
                          若为 None，则使用默认路径
        :return: 是否全部成功
        """
        if not self.is_flying:
            print("错误：无人机未起飞，请先调用 takeoff_and_hover()")
            return False

        if waypoints is None:
            waypoints = [
                (20, 0, -10),
                (20, 15, -10),
                (0, 15, -12),
                (0, 0, -10),
            ]

        print("开始探索废弃公园...")
        for idx, (x, y, z) in enumerate(waypoints, 1):
            print(f"[{idx}/{len(waypoints)}] 飞往位置: ({x}, {y}, {z})")
            try:
                self.client.moveToPositionAsync(x, y, z, MOVE_SPEED).join()
                time.sleep(POST_MOVE_DELAY)
                self.capture_park_image()
            except Exception as e:
                print(f"✗ 移动失败: {e}")
                return False

        print("探索完成！")
        return True

    def cleanup(self) -> None:
        """清理资源：降落、锁定无人机、禁用 API 控制"""
        if not self.is_connected:
            return

        print("正在降落...")
        try:
            if self.is_flying:
                self.client.landAsync().join()
                self.is_flying = False
            if self.is_armed:
                self.client.armDisarm(False)
                self.is_armed = False
            self.client.enableApiControl(False)
            print("✓ 无人机已降落并锁定")
        except Exception as e:
            print(f"✗ 清理过程中出错: {e}")


# ----------------------------------------------------------------------
# 交互式测试脚本（当直接运行此文件时执行）
# ----------------------------------------------------------------------
def interactive_test():
    """交互式测试菜单"""
    print("=== AbandonedPark 无人机测试 ===")
    print("请确保 AbandonedPark.exe 已运行并按 F2 切换到无人机模式。")
    input("按回车键继续...")

    # 创建控制器对象（自动连接）
    sim = AbandonedParkSimulator()

    if not sim.is_connected:
        print("无法连接，测试终止。")
        return

    try:
        while True:
            print("\n" + "-" * 40)
            print("请选择操作：")
            print("1. 切换到无人机模式（解锁）")
            print("2. 起飞并悬停")
            print("3. 捕获图像")
            print("4. 探索公园（默认路径）")
            print("5. 降落并清理")
            print("0. 退出")
            choice = input("请输入数字: ").strip()

            if choice == '1':
                sim.ensure_drone_mode()
            elif choice == '2':
                alt = input("请输入高度（米，默认10）: ").strip()
                alt = float(alt) if alt else 10.0
                sim.takeoff_and_hover(alt)
            elif choice == '3':
                sim.capture_park_image()
            elif choice == '4':
                sim.explore_park()
            elif choice == '5':
                sim.cleanup()
            elif choice == '0':
                break
            else:
                print("无效输入，请重新选择。")
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        sim.cleanup()
        print("测试结束。")


if __name__ == "__main__":
    interactive_test()