import random
import time


class CarlaSpeedAlarmSystem:
    """Carla无人车速度报警系统"""

    def __init__(self):
        # 速度区间配置（可扩展）
        self.speed_ranges = [
            (0, 60, "正常"),
            (61, 120, "超速"),
            (121, 150, "超速危险")
        ]

    def get_carla_vehicle_speed(self):
        """
        模拟从Carla获取车辆实时速度（km/h）
        实际项目中替换为Carla API的速度读取逻辑：
        vehicle.get_velocity() → 转换为km/h
        """
        # 模拟速度波动：0~160（包含异常值）
        return random.uniform(0, 160)

    def judge_speed_status(self, speed):
        """根据速度判断状态"""
        # 检查速度是否在合法范围外
        if speed < 0 or speed > 150:
            return "速度异常"

        # 遍历速度区间判断状态
        for min_speed, max_speed, status in self.speed_ranges:
            if min_speed <= speed <= max_speed:
                return status
        return "未知状态"

    def run(self):
        """运行速度报警系统"""
        print("=== Carla无人车速度报警系统启动 ===")
        print("速度区间说明：")
        print("0~60 km/h → 正常")
        print("61~120 km/h → 超速")
        print("121~150 km/h → 超速危险")
        print("-" * 50)

        try:
            while True:
                # 获取速度
                current_speed = self.get_carla_vehicle_speed()
                # 判断状态
                speed_status = self.judge_speed_status(current_speed)
                # 输出控制面板信息
                print(f"当前车速：{current_speed:.1f} km/h | 状态：{speed_status}")
                # 模拟实时更新（每1秒刷新一次）
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n=== 系统已停止运行 ===")


if __name__ == "__main__":
    # 实例化并运行系统
    alarm_system = CarlaSpeedAlarmSystem()
    alarm_system.run()