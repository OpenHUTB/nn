"""
简单的车辆状态监控演示
"""
import time


class VehicleMonitor:
    """车辆状态监控类"""

    def __init__(self, vehicle_name="TestVehicle"):
        self.vehicle_name = vehicle_name
        self.speed = 0.0  # km/h
        self.position = [0.0, 0.0]  # x, y
        self.fuel = 100.0  # 百分比

    def update_speed(self, new_speed):
        """更新速度"""
        self.speed = new_speed
        print(f"[{self.vehicle_name}] 速度更新: {self.speed:.1f} km/h")

    def update_position(self, x, y):
        """更新位置"""
        self.position = [x, y]
        print(f"[{self.vehicle_name}] 位置更新: ({x:.2f}, {y:.2f})")

    def consume_fuel(self, amount):
        """消耗燃料"""
        self.fuel = max(0, self.fuel - amount)
        print(f"[{self.vehicle_name}] 燃料剩余: {self.fuel:.1f}%")

    def get_status(self):
        """获取车辆状态"""
        return {
            "车辆名称": self.vehicle_name,
            "当前速度": f"{self.speed:.1f} km/h",
            "当前位置": f"({self.position[0]:.2f}, {self.position[1]:.2f})",
            "燃料剩余": f"{self.fuel:.1f}%"
        }

    def display_status(self):
        """显示车辆状态"""
        status = self.get_status()
        print("\n" + "=" * 40)
        print("车辆状态报告")
        print("=" * 40)
        for key, value in status.items():
            print(f"{key}: {value}")
        print("=" * 40 + "\n")


def main():
    """主函数 - 演示车辆监控功能"""
    print("启动车辆监控系统...\n")

    # 创建车辆监控器
    vehicle = VehicleMonitor("DemoCar-001")

    # 模拟车辆运行
    for i in range(5):
        print(f"\n--- 第 {i+1} 次更新 ---")

        # 更新速度
        vehicle.update_speed(20 + i * 10)

        # 更新位置
        vehicle.update_position(i * 5, i * 3)

        # 消耗燃料
        vehicle.consume_fuel(5)

        # 显示状态
        vehicle.display_status()

        time.sleep(0.5)

    print("车辆监控演示完成！")


if __name__ == "__main__":
    main()
