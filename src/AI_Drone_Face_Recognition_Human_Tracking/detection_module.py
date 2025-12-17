# -*- coding: utf-8 -*-
"""
无人机检测模块
功能：
1. 独立运行：模拟检测无人机状态、环境障碍物
2. 被主程序导入：提供检测接口，返回预警信息
"""
import numpy as np
import random
from enum import Enum
import time


# ===================== 检测类型枚举（便于主程序调用） =====================
class DetectionType(Enum):
    OBSTACLE = "障碍物检测"
    BATTERY = "电量检测"
    POSITION = "位置检测"
    COLLISION = "碰撞预警"
    STATE = "状态检测"


# ===================== 核心检测类 =====================
class DroneDetection:
    def __init__(self, drone=None):
        """
        初始化检测模块
        :param drone: 可选，传入虚拟无人机对象（主程序调用时传）
        """
        self.drone = drone  # 关联无人机对象
        self.obstacle_list = self._generate_obstacles()  # 生成模拟障碍物
        self.warning_threshold = {
            "battery": 20.0,  # 电量预警阈值（%）
            "height": 0.5,  # 高度过低预警阈值（m）
            "distance": 1.0  # 障碍物距离预警阈值（m）
        }

    def _generate_obstacles(self):
        """生成模拟环境障碍物（随机坐标）"""
        obstacles = []
        for _ in range(10):  # 生成10个随机障碍物
            x = random.uniform(-10, 10)
            y = random.uniform(-10, 10)
            z = random.uniform(0, 8)
            obstacles.append(np.array([x, y, z]))
        return obstacles

    def detect_obstacle(self):
        """
        障碍物检测：计算无人机与最近障碍物的距离
        :return: dict - 检测结果（距离、是否预警、最近障碍物坐标）
        """
        if self.drone is None:
            return {
                "type": DetectionType.OBSTACLE.value,
                "status": "未关联无人机",
                "distance": None,
                "warning": False,
                "nearest_obstacle": None
            }

        # 计算无人机与每个障碍物的欧氏距离
        drone_pos = self.drone.position
        distances = [np.linalg.norm(drone_pos - obs) for obs in self.obstacle_list]
        min_distance = min(distances)
        nearest_idx = np.argmin(distances)
        nearest_obs = self.obstacle_list[nearest_idx]

        # 判断是否触发预警
        warning = min_distance < self.warning_threshold["distance"]

        return {
            "type": DetectionType.OBSTACLE.value,
            "status": "检测完成",
            "distance": round(min_distance, 2),
            "warning": warning,
            "nearest_obstacle": nearest_obs.round(2),
            "message": f"⚠️ 距离障碍物仅{min_distance:.2f}m，请注意避让！" if warning else "✅ 无障碍物风险"
        }

    def detect_battery(self):
        """
        电量检测：判断是否低于预警阈值
        :return: dict - 检测结果
        """
        if self.drone is None:
            return {
                "type": DetectionType.BATTERY.value,
                "status": "未关联无人机",
                "battery": None,
                "warning": False,
                "message": "未关联无人机，无法检测电量"
            }

        battery = self.drone.battery
        warning = battery < self.warning_threshold["battery"]

        return {
            "type": DetectionType.BATTERY.value,
            "status": "检测完成",
            "battery": round(battery, 2),
            "warning": warning,
            "message": f"⚠️ 电量低（{battery:.1f}%），请尽快返航！" if warning else f"✅ 电量充足（{battery:.1f}%）"
        }

    def detect_position(self):
        """
        位置检测：判断高度是否过低/超出边界
        :return: dict - 检测结果
        """
        if self.drone is None:
            return {
                "type": DetectionType.POSITION.value,
                "status": "未关联无人机",
                "position": None,
                "warning": False,
                "message": "未关联无人机，无法检测位置"
            }

        pos = self.drone.position
        # 兼容主程序枚举状态和独立运行字符串状态
        drone_state = self.drone.state.value if hasattr(self.drone.state, "value") else self.drone.state
        height_warning = pos[2] < self.warning_threshold["height"] and drone_state == "FLYING"
        boundary_warning = abs(pos[0]) > 15 or abs(pos[1]) > 15  # 水平边界±15m

        warning = height_warning or boundary_warning
        messages = []
        if height_warning:
            messages.append(f"⚠️ 飞行高度过低（{pos[2]:.1f}m），请注意！")
        if boundary_warning:
            messages.append(f"⚠️ 超出安全边界（坐标：{pos[:2].round(1)}），请返航！")
        if not warning:
            messages.append(f"✅ 位置正常（{pos.round(1)}）")

        return {
            "type": DetectionType.POSITION.value,
            "status": "检测完成",
            "position": pos.round(2),
            "warning": warning,
            "message": " | ".join(messages)
        }

    def detect_collision(self):
        """
        碰撞预警：预测未来1秒是否有碰撞风险
        :return: dict - 检测结果
        """
        if self.drone is None:
            return {
                "type": DetectionType.COLLISION.value,
                "status": "未关联无人机",
                "risk": False,
                "message": "未关联无人机，无法预测碰撞风险"
            }

        # 预测1秒后无人机位置（基于当前速度）
        future_pos = self.drone.position + self.drone.velocity * 1.0
        # 计算与障碍物的距离
        distances = [np.linalg.norm(future_pos - obs) for obs in self.obstacle_list]
        min_future_dist = min(distances)
        risk = min_future_dist < 0.5  # 距离<0.5m判定为碰撞风险

        return {
            "type": DetectionType.COLLISION.value,
            "status": "检测完成",
            "risk": risk,
            "future_position": future_pos.round(2),
            "message": "🚨 1秒后有碰撞风险！请立即调整方向！" if risk else "✅ 无碰撞风险"
        }

    def detect_state(self):
        """
        状态检测：判断无人机当前状态是否正常
        :return: dict - 检测结果
        """
        if self.drone is None:
            return {
                "type": DetectionType.STATE.value,
                "status": "未关联无人机",
                "drone_state": None,
                "message": "未关联无人机，无法检测状态"
            }

        # 兼容主程序枚举状态（DroneState）和独立运行字符串状态
        state = self.drone.state.value if hasattr(self.drone.state, "value") else self.drone.state
        if state == "LANDED":
            message = "✅ 无人机处于落地状态，状态正常"
        elif state == "FLYING" and self.drone.battery > 10:
            message = "✅ 无人机处于飞行状态，状态正常"
        else:
            message = f"⚠️ 无人机飞行状态异常（电量{self.drone.battery:.1f}%）"

        return {
            "type": DetectionType.STATE.value,
            "status": "检测完成",
            "drone_state": state,
            "warning": state == "FLYING" and self.drone.battery <= 10,
            "message": message
        }

    def full_detection(self):
        """
        全量检测：执行所有检测项
        :return: list - 所有检测结果
        """
        results = [
            self.detect_state(),
            self.detect_battery(),
            self.detect_position(),
            self.detect_obstacle(),
            self.detect_collision()
        ]
        return results


# ===================== 独立运行测试代码 =====================
def main():
    """模块独立运行时的测试逻辑"""
    print("=" * 60)
    print("📊 无人机检测模块 - 独立测试模式")
    print("=" * 60)

    # 模拟无人机对象（独立运行时使用，简化状态为字符串，避免枚举类错误）
    class MockDrone:
        def __init__(self):
            self.position = np.array([2.0, 1.5, 1.0])  # 模拟位置
            self.velocity = np.array([0.8, 0.5, 0.0])  # 模拟速度
            self.state = "FLYING"  # 直接用字符串标识状态，替代枚举类
            self.battery = 18.5  # 模拟电量

    # 初始化检测模块
    mock_drone = MockDrone()
    detector = DroneDetection(drone=mock_drone)

    # 循环检测（模拟实时监测）
    try:
        while True:
            print(f"\n⏰ 检测时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
            # 执行全量检测
            all_results = detector.full_detection()
            for res in all_results:
                print(f"[{res['type']}] {res['message']}")

            # 模拟无人机位置变化
            mock_drone.position += mock_drone.velocity * 0.5
            mock_drone.battery -= 0.5  # 模拟电量消耗
            # 模拟飞行状态切换（每10秒落地一次）
            if mock_drone.battery < 10:
                mock_drone.state = "LANDED"
            time.sleep(2)  # 每2秒检测一次

    except KeyboardInterrupt:
        print("\n\n🛑 检测模块独立测试已停止")


# 仅当模块独立运行时执行测试
if __name__ == "__main__":
    main()