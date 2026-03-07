#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
airsim_connection.py - AbandonedPark 模拟器无人机控制模块
提供连接、解锁、起飞、图像捕获、路径探索和降落等功能。
"""

import airsim
import time
import os
import sys
import cv2
import numpy as np
from datetime import datetime


class AbandonedParkSimulator:
    """废弃公园场景的无人机模拟器控制类"""

    def __init__(self, ip="127.0.0.1", port=41451, auto_connect=True):
        """
        初始化控制器
        :param ip: 模拟器 IP 地址
        :param port: 通信端口
        :param auto_connect: 是否自动连接
        """
        self.ip = ip
        self.port = port
        self.client = None
        self.connected = False
        self.flying = False
        self.image_save_dir = None  # 图像保存目录，由 capture 方法自动创建

        if auto_connect:
            self.connect()

    def connect(self, max_retries=3, retry_delay=2):
        """
        连接到 AirSim 模拟器
        :param max_retries: 最大重试次数
        :param retry_delay: 重试间隔（秒）
        :return: 是否连接成功
        """
        print(f"正在连接到 AbandonedPark 模拟器 {self.ip}:{self.port}...")
        for attempt in range(1, max_retries + 1):
            try:
                self.client = airsim.MultirotorClient(ip=self.ip, port=self.port)
                self.client.confirmConnection()
                # 使用 ping 验证连接
                ping_time = self.client.ping()
                print(f"✓ 连接成功！响应时间: {ping_time} ms (尝试 {attempt})")
                self.connected = True
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
        if not self.connected:
            print("错误：未连接到模拟器")
            return False

        print("切换到无人机模式...")
        try:
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            print("✓ 无人机已解锁")
            return True
        except Exception as e:
            print(f"✗ 切换模式失败: {e}")
            print("提示：请确保模拟器中已选择无人机模式（按 F2 切换）")
            return False

    def takeoff_and_hover(self, altitude=10, timeout=10):
        """
        起飞到指定高度并悬停
        :param altitude: 目标高度（米）
        :param timeout: 起飞超时时间（秒）
        """
        if not self.connected:
            print("错误：未连接到模拟器")
            return False

        print(f"起飞到 {altitude} 米高度...")
        try:
            self.client.takeoffAsync().join(timeout=timeout)
            time.sleep(2)
            self.client.moveToZAsync(-altitude, 3).join(timeout=timeout)
            time.sleep(1)
            self.flying = True
            print(f"✓ 已在 {altitude} 米高度悬停")
            return True
        except Exception as e:
            print(f"✗ 起飞失败: {e}")
            self.flying = False
            return False

    def capture_park_image(self, camera_name="0", save_dir=None):
        """
        从指定摄像头捕获图像并保存
        :param camera_name: 摄像头名称，默认为前置 "0"
        :param save_dir: 保存目录，若为 None 则自动创建带时间戳的目录
        :return: 图像 numpy 数组，失败返回 None
        """
        if not self.connected:
            print("错误：未连接到模拟器")
            return None

        # 设置保存目录
        if save_dir is None:
            if self.image_save_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.image_save_dir = f"captures_{timestamp}"
            save_dir = self.image_save_dir
        os.makedirs(save_dir, exist_ok=True)

        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
            ])
            if not responses:
                print("✗ 未收到图像响应")
                return None

            response = responses[0]
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)

            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = os.path.join(save_dir, f"capture_{timestamp}.jpg")
            cv2.imwrite(filename, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            print(f"✓ 图像已保存: {filename}")
            return img_rgb
        except Exception as e:
            print(f"✗ 图像捕获失败: {e}")
            return None

    def explore_park(self, waypoints=None):
        """
        执行路径探索：依次飞往航点并在每个航点拍照
        :param waypoints: 航点列表，每个元素为 (x, y, z) 元组，z 为负值
                          若为 None，则使用默认路径
        """
        if not self.flying:
            print("错误：无人机尚未起飞")
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
                self.client.moveToPositionAsync(x, y, z, 3).join()
                self.capture_park_image()
                time.sleep(1)
            except Exception as e:
                print(f"✗ 移动失败: {e}")
                return False
        print("探索完成！")
        return True

    def land(self):
        """降落无人机并清理资源"""
        if not self.connected:
            return

        print("正在降落...")
        try:
            if self.flying:
                self.client.landAsync().join()
                self.flying = False
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            print("✓ 无人机已降落并锁定")
        except Exception as e:
            print(f"✗ 降落过程中出错: {e}")

    def cleanup(self):
        """清理资源（同 land，作为别名）"""
        self.land()


# ----------------------------------------------------------------------
# 交互式测试脚本
# ----------------------------------------------------------------------
def run_test():
    """运行交互式测试"""
    print("=== AbandonedPark 无人机测试 ===")
    print("请确保 AbandonedPark.exe 已运行并按 F2 切换到无人机模式。")
    input("按回车键继续...")

    sim = AbandonedParkSimulator(auto_connect=True)
    if not sim.connected:
        print("无法连接，退出测试。")
        return

    try:
        while True:
            print("\n" + "-" * 40)
            print("请选择测试项：")
            print("1. 切换无人机模式（解锁）")
            print("2. 起飞并悬停")
            print("3. 捕获图像")
            print("4. 探索公园（默认路径）")
            print("5. 降落")
            print("0. 退出")
            choice = input("请输入数字: ").strip()

            if choice == '1':
                sim.ensure_drone_mode()
            elif choice == '2':
                alt = input("请输入高度（米，默认10）: ").strip()
                alt = int(alt) if alt.isdigit() else 10
                sim.takeoff_and_hover(alt)
            elif choice == '3':
                sim.capture_park_image()
            elif choice == '4':
                if not sim.flying:
                    print("请先起飞！")
                else:
                    sim.explore_park()
            elif choice == '5':
                sim.land()
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
    run_test()