#!/usr/bin/env python3
"""
无人机状态检查工具
用于连接 AirSim 模拟器，获取并显示无人机状态，支持可选的解锁操作。
"""

import argparse
import time
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

import airsim

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DroneStatus:
    """无人机状态数据类"""
    position_x: float
    position_y: float
    position_z: float
    speed: float
    battery: float
    collided: bool
    timestamp: float

    @classmethod
    def from_multirotor_state(cls, state: airsim.MultirotorState) -> 'DroneStatus':
        pos = state.kinematics_estimated.position
        return cls(
            position_x=pos.x_val,
            position_y=pos.y_val,
            position_z=pos.z_val,
            speed=state.speed,
            battery=state.battery,
            collided=state.collision.has_collided,
            timestamp=time.time()
        )

    def display(self) -> None:
        """打印状态信息"""
        print(f"当前位置: X={self.position_x:.2f}, Y={self.position_y:.2f}, Z={self.position_z:.2f}")
        print(f"当前速度: {self.speed:.2f} m/s")
        print(f"电池电量: {self.battery:.1f}%")
        print(f"是否碰撞: {'是' if self.collided else '否'}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='检查 AirSim 无人机状态')
    parser.add_argument('--unlock', action='store_true', help='是否解锁无人机')
    parser.add_argument('--retries', type=int, default=3, help='连接重试次数')
    parser.add_argument('--timeout', type=float, default=5.0, help='连接超时（秒）')
    parser.add_argument('--save', type=str, help='将状态保存到指定JSON文件')
    parser.add_argument('--no-color', action='store_true', help='禁用彩色输出')
    return parser.parse_args()


def connect_simulator(retries: int, timeout: float) -> Optional[airsim.MultirotorClient]:
    """连接到模拟器，支持重试和超时"""
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"连接尝试 {attempt}/{retries}...")
            client = airsim.MultirotorClient()
            # 设置超时（AirSim Python客户端可能不支持直接超时，但 confirmConnection 有内部超时）
            client.confirmConnection()
            if client.ping():
                logger.info("连接成功")
                return client
        except Exception as e:
            logger.warning(f"连接失败: {e}")
            if attempt < retries:
                time.sleep(1)
    logger.error("无法连接到模拟器，请确保 AbandonedPark.exe 已运行")
    return None


def enable_drone_control(client: airsim.MultirotorClient) -> bool:
    """启用API控制并解锁"""
    try:
        logger.info("启用API控制...")
        client.enableApiControl(True)
        logger.info("解锁无人机...")
        client.armDisarm(True)
        return True
    except Exception as e:
        logger.error(f"解锁失败: {e}")
        return False


def get_status(client: airsim.MultirotorClient) -> Optional[DroneStatus]:
    """获取并返回无人机状态"""
    try:
        state = client.getMultirotorState()
        return DroneStatus.from_multirotor_state(state)
    except Exception as e:
        logger.error(f"获取状态失败: {e}")
        return None


def save_status_to_file(status: DroneStatus, filepath: str) -> None:
    """将状态保存为JSON文件"""
    try:
        data = asdict(status)
        data['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data['timestamp']))
        Path(filepath).write_text(json.dumps(data, indent=2), encoding='utf-8')
        logger.info(f"状态已保存到 {filepath}")
    except Exception as e:
        logger.error(f"保存文件失败: {e}")


def main():
    args = parse_args()

    print("=" * 50)
    print("无人机状态检查")
    print("=" * 50)

    # 连接模拟器
    client = connect_simulator(args.retries, args.timeout)
    if not client:
        return

    print("✓ 已连接到 AbandonedPark 模拟器")

    # 获取状态
    status = get_status(client)
    if status:
        status.display()
        if args.save:
            save_status_to_file(status, args.save)
    else:
        print("⚠️ 无法获取完整状态信息")

    # 可选解锁
    if args.unlock:
        print("\n准备解锁无人机...")
        if enable_drone_control(client):
            print("✓ 无人机已解锁，准备就绪")
        else:
            print("❌ 解锁失败，请检查模拟器是否处于无人机模式")

    print("\n" + "=" * 50)
    print("状态检查完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()