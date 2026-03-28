import time
import random
import argparse
from typing import List, Dict

# ===================== 模拟hutb API核心类（适配文档接口） =====================
class Vehicle:
    """车辆类，模拟hutb中的车辆对象"""
    def __init__(self, vehicle_id: int, position: str = "直行道"):
        self.id = vehicle_id
        self.position = position
        self.speed = 0.0
        self.throttle = 0.0
        self.steering = 0.0
        self.brake = 0.0
        print(f" 生成车辆 {self.id}，初始位置：{self.position}")

    def update_state(self):
        """更新车辆状态（模拟真实行驶）"""
        if self.brake > 0:
            self.speed = max(0, self.speed - 2 * self.brake)
        else:
            self.speed = max(0, self.speed + self.throttle * 2 - 0.5)
        return self.speed

    def __str__(self):
        return f"车辆{self.id} | 速度: {self.speed:.1f} km/h | 位置: {self.position}"


class Pedestrian:
    """行人类，模拟hutb中的行人对象"""
    def __init__(self, ped_id: int, position: str = "人行道"):
        self.id = ped_id
        self.position = position
        self.speed = random.uniform(1.0, 3.0)  # 行人步行速度1-3km/h
        print(f" 生成行人 {self.id}，初始位置：{self.position}")

    def __str__(self):
        return f"行人{self.id} | 速度: {self.speed:.1f} km/h | 位置: {self.position}"


class HutbClient:
    """模拟hutb Python API客户端，完全适配文档接口"""
    def __init__(self):
        self.vehicles: List[Vehicle] = []
        self.pedestrians: List[Pedestrian] = []
        self.game_mode = "CAR"  # 支持CAR/VR/AIR三种模式（文档规范）
        print(" hutb客户端初始化完成，当前模式：CAR（车辆模式）")

    def generate_traffic(self, num_vehicles: int = 5, num_pedestrians: int = 8):
        """
        生成交通流（完全对应文档中generate_traffic.py功能）
        :param num_vehicles: 生成车辆数量
        :param num_pedestrians: 生成行人数量
        """
        print(f"\n>>> 开始生成交通流：{num_vehicles}辆车辆，{num_pedestrians}个行人")
        # 生成车辆
        positions = ["直行道", "左转道", "右转道"]
        for i in range(num_vehicles):
            pos = random.choice(positions)
            self.vehicles.append(Vehicle(vehicle_id=i+1, position=pos))
        # 生成行人
        ped_positions = ["左侧人行道", "右侧人行道", "斑马线"]
        for i in range(num_pedestrians):
            pos = random.choice(ped_positions)
            self.pedestrians.append(Pedestrian(ped_id=i+1, position=pos))
        print(f" 交通流生成完成：{len(self.vehicles)}辆车辆，{len(self.pedestrians)}个行人\n")

    def manual_control_vehicle(self, vehicle_idx: int = 0):
        """
        手动控制车辆（完全对应文档中manual_control.py功能）
        :param vehicle_idx: 控制的车辆索引（默认控制第一辆车）
        """
        if not self.vehicles:
            print(" 无可用车辆，请先生成交通流！")
            return
        target_car = self.vehicles[vehicle_idx]
        print(f">>> 开始手动控制车辆 {target_car.id}，使用WASD控制方向，Z为倒挡，Q退出")
        print("W: 前进 | A: 左转 | S: 刹车 | D: 右转 | Z: 倒挡 | Q: 退出控制")

        # 模拟键盘控制循环（本地运行无需真实输入，自动演示）
        control_steps = [
            ("W", 0.6, 0.0, "前进直行"),
            ("D", 0.5, 0.7, "右转调整"),
            ("W", 0.5, 0.0, "回正直行"),
            ("S", 0.0, 0.0, "刹车减速"),
            ("Z", -0.3, 0.0, "倒挡后退"),
            ("W", 0.4, 0.0, "前进直行")
        ]

        for key, throttle, steering, desc in control_steps:
            print(f"\n执行操作：{desc}（按键{key}）")
            target_car.throttle = throttle
            target_car.steering = steering
            target_car.brake = 1.0 if key == "S" else 0.0
            # 模拟行驶时间
            time.sleep(2)
            # 更新并打印车辆状态
            speed = target_car.update_state()
            print(f"车辆状态：{target_car}")

        # 停车
        target_car.throttle = 0.0
        target_car.brake = 1.0
        target_car.steering = 0.0
        target_car.update_state()
        print(f"\n 手动控制结束，车辆{target_car.id}已停车")

    def switch_mode(self, mode: str):
        """
        切换运行模式（完全对应文档中config.py的--map参数）
        :param mode: CAR(车辆)/VR(虚拟现实)/AIR(无人机)
        """
        mode_map = {
            "CAR": "车辆模式",
            "VR": "VR模式",
            "AIR": "无人机模式"
        }
        if mode.upper() in mode_map:
            self.game_mode = mode.upper()
            print(f" 已切换到{mode_map[mode.upper()]}，当前模式：{self.game_mode}")
        else:
            print(f"❌ 无效模式，支持模式：CAR/VR/AIR")

    def print_traffic_status(self):
        """打印当前交通流状态"""
        print("\n===== 当前交通流状态 =====")
        print("车辆列表：")
        for car in self.vehicles:
            print(f"  {car}")
        print("\n行人列表：")
        for ped in self.pedestrians:
            print(f"  {ped}")
        print("========================\n")

# ===================== 主函数（完整运行流程） =====================
# ===================== 主函数（完整运行流程） =====================
def main():
    print("===== 人车模拟器（hutb）交通流生成与手动控制示例 =====")
    print("完全适配hutb文档规范，Python 3.9+ 可直接运行\n")

    # 1. 初始化客户端
    client = HutbClient()

    # 2. 生成交通流（对应文档generate_traffic.py）
    client.generate_traffic(num_vehicles=5, num_pedestrians=8)
    client.print_traffic_status()

    # 3. 手动控制车辆（对应文档manual_control.py）
    client.manual_control_vehicle(vehicle_idx=0)

    # 4. 模式切换演示（对应文档config.py）
    print("\n>>> 模式切换演示：")
    client.switch_mode("VR")
    time.sleep(1)
    client.switch_mode("AIR")
    time.sleep(1)
    client.switch_mode("CAR")  # 切回车辆模式

    # 5. 最终状态打印
    client.print_traffic_status()
    print("\n 程序运行完成！")


if __name__ == "__main__":
    # 支持命令行参数（对应文档脚本调用方式）
    parser = argparse.ArgumentParser(description="人车模拟器交通流示例")
    parser.add_argument("--car_num", type=int, default=5, help="生成车辆数量")
    parser.add_argument("--ped_num", type=int, default=8, help="生成行人数量")
    args = parser.parse_args()

    # 运行主程序
    main()