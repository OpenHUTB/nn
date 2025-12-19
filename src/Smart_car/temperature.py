import time
import random
from enum import Enum

# 定义系统常量
DEFAULT_TARGET_TEMP = 24  # 默认目标温度(℃)
TEMP_TOLERANCE = 1  # 温度容差(℃)
MAX_FAN_SPEED = 5  # 最大风速档
MIN_FAN_SPEED = 1  # 最小风速档


# 空调模式枚举
class AC_Mode(Enum):
    COOL = "制冷"
    HEAT = "制热"
    VENT = "通风"
    OFF = "关闭"


# 运行模式枚举
class Run_Mode(Enum):
    AUTO = "自动"
    MANUAL = "手动"


class TemperatureSensor:
    """温度传感器类 - 模拟采集车内/车外温度"""

    def __init__(self):
        self.interior_temp = 25  # 初始车内温度
        self.exterior_temp = 30  # 初始车外温度

    def read_temperatures(self):
        """模拟读取温度（加入微小随机波动）"""
        self.interior_temp += random.uniform(-0.5, 0.5)
        self.exterior_temp += random.uniform(-0.8, 0.8)
        # 温度边界限制
        self.interior_temp = max(10, min(45, self.interior_temp))
        self.exterior_temp = max(-20, min(50, self.exterior_temp))
        return round(self.interior_temp, 1), round(self.exterior_temp, 1)


class SunlightSensor:
    """阳光强度传感器 - 影响车内温度变化"""

    def read_intensity(self):
        """返回0-10的阳光强度值"""
        return round(random.uniform(0, 10), 1)


class PassengerDetector:
    """乘客检测 - 模拟检测车内乘客数量"""

    def get_passenger_count(self):
        """返回0-5的乘客数"""
        return random.randint(0, 5)


class AirConditioner:
    """空调执行器类 - 控制空调运行"""

    def __init__(self):
        self.ac_mode = AC_Mode.OFF
        self.fan_speed = MIN_FAN_SPEED
        self.target_temp = DEFAULT_TARGET_TEMP
        self.run_mode = Run_Mode.AUTO

    def set_mode(self, mode):
        """设置空调模式"""
        if isinstance(mode, AC_Mode):
            self.ac_mode = mode
            print(f"空调模式已切换为: {self.ac_mode.value}")

    def set_fan_speed(self, speed):
        """设置风速（1-5档）"""
        if MIN_FAN_SPEED <= speed <= MAX_FAN_SPEED:
            self.fan_speed = speed
            print(f"风速已设置为: {self.fan_speed}档")

    def set_target_temp(self, temp):
        """设置目标温度（16-30℃）"""
        if 16 <= temp <= 30:
            self.target_temp = temp
            print(f"目标温度已设置为: {self.target_temp}℃")

    def set_run_mode(self, mode):
        """设置运行模式（自动/手动）"""
        if isinstance(mode, Run_Mode):
            self.run_mode = mode
            print(f"运行模式已切换为: {self.run_mode.value}")


class TemperatureControlSystem:
    """无人车温度调节核心系统"""

    def __init__(self):
        # 初始化传感器和执行器
        self.temp_sensor = TemperatureSensor()
        self.sun_sensor = SunlightSensor()
        self.passenger_detector = PassengerDetector()
        self.aircon = AirConditioner()

    def calculate_adjustment(self):
        """核心算法：根据环境参数计算空调调节策略"""
        interior_temp, exterior_temp = self.temp_sensor.read_temperatures()
        sunlight_intensity = self.sun_sensor.read_intensity()
        passenger_count = self.passenger_detector.get_passenger_count()

        # 打印当前环境参数
        print(f"\n=== 环境参数 ===")
        print(f"车内温度: {interior_temp}℃")
        print(f"车外温度: {exterior_temp}℃")
        print(f"阳光强度: {sunlight_intensity}")
        print(f"乘客数量: {passenger_count}")

        # 目标温度动态调整（乘客越多/阳光越强，目标温度略低）
        base_target = DEFAULT_TARGET_TEMP
        dynamic_target = base_target - (passenger_count * 0.5) - (sunlight_intensity * 0.2)
        dynamic_target = max(18, min(26, dynamic_target))  # 限制在18-26℃

        # 自动模式下的调节逻辑
        if self.aircon.run_mode == Run_Mode.AUTO:
            # 温度偏差计算
            temp_diff = interior_temp - dynamic_target

            # 制冷逻辑
            if temp_diff > TEMP_TOLERANCE:
                self.aircon.set_mode(AC_Mode.COOL)
                # 温差越大，风速越高
                fan_speed = MIN_FAN_SPEED + min(int(temp_diff), MAX_FAN_SPEED - MIN_FAN_SPEED)
                self.aircon.set_fan_speed(fan_speed)
                self.aircon.set_target_temp(dynamic_target)

            # 制热逻辑
            elif temp_diff < -TEMP_TOLERANCE:
                self.aircon.set_mode(AC_Mode.HEAT)
                fan_speed = MIN_FAN_SPEED + min(int(abs(temp_diff)), MAX_FAN_SPEED - MIN_FAN_SPEED)
                self.aircon.set_fan_speed(fan_speed)
                self.aircon.set_target_temp(dynamic_target)

            # 温度适宜 - 通风模式
            else:
                self.aircon.set_mode(AC_Mode.VENT)
                self.aircon.set_fan_speed(MIN_FAN_SPEED)

        # 打印当前空调状态
        print(f"\n=== 空调状态 ===")
        print(f"运行模式: {self.aircon.run_mode.value}")
        print(f"空调模式: {self.aircon.ac_mode.value}")
        print(f"目标温度: {self.aircon.target_temp}℃")
        print(f"当前风速: {self.aircon.fan_speed}档")

    def manual_control(self, mode, target_temp=None, fan_speed=None):
        """手动控制接口"""
        if self.aircon.run_mode != Run_Mode.MANUAL:
            self.aircon.set_run_mode(Run_Mode.MANUAL)

        self.aircon.set_mode(mode)
        if target_temp:
            self.aircon.set_target_temp(target_temp)
        if fan_speed:
            self.aircon.set_fan_speed(fan_speed)

    def run(self, duration=10):
        """系统主运行函数"""
        print("无人车温度调节系统启动...")
        print(f"系统将运行 {duration} 秒，自动调节温度")

        start_time = time.time()
        while time.time() - start_time < duration:
            self.calculate_adjustment()
            time.sleep(2)  # 每2秒调节一次

        print("\n系统运行结束")


# 测试代码
if __name__ == "__main__":
    # 初始化系统
    temp_system = TemperatureControlSystem()

    # 示例1：自动模式运行10秒
    temp_system.run(duration=10)

    # 示例2：切换到手动模式，设置制冷22℃，3档风速
    print("\n--- 切换到手动模式 ---")
    temp_system.manual_control(
        mode=AC_Mode.COOL,
        target_temp=22,
        fan_speed=3
    )

    # 手动模式下继续运行5秒
    time.sleep(5)
    temp_system.calculate_adjustment()