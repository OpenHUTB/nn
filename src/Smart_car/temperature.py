import time
import random
from enum import Enum

# 定义温度调节模式枚举
class TempMode(Enum):
    AUTO = "自动模式"    # 自动根据目标温度调节
    COOL = "制冷模式"    # 仅制冷
    HEAT = "制热模式"    # 仅制热
    FAN = "仅吹风模式"   # 仅通风，不控温
    OFF = "关闭模式"     # 系统关闭

# 定义温度调节系统类
class AutoCarTempSystem:
    def __init__(self):
        # 系统基础配置
        self.target_temp = 25.0          # 目标温度(℃)，默认25℃
        self.current_temp = 25.0         # 当前温度(℃)，初始默认值
        self.mode = TempMode.AUTO        # 初始模式：自动
        self.fan_speed = 2               # 风扇转速(1-5档)，默认2档
        self.is_running = True           # 系统运行状态
        self.temp_tolerance = 0.5        # 温度容差(℃)，避免频繁启停
        self.max_temp = 45.0             # 最高安全温度
        self.min_temp = 5.0              # 最低安全温度

    def simulate_temp_sensor(self):
        """模拟温度传感器读取当前温度（含微小波动）"""
        # 模拟真实环境温度波动 ±0.3℃
        fluctuation = random.uniform(-0.3, 0.3)
        self.current_temp += fluctuation
        # 限制温度在安全范围内
        self.current_temp = max(self.min_temp, min(self.max_temp, self.current_temp))
        return round(self.current_temp, 1)

    def set_target_temp(self, temp):
        """设置目标温度（含合法性校验）"""
        if self.min_temp <= temp <= self.max_temp:
            self.target_temp = temp
            print(f"✅ 目标温度已设置为：{temp}℃")
        else:
            print(f"❌ 温度设置失败！请设置{self.min_temp}~{self.max_temp}℃范围内的温度")

    def set_mode(self, new_mode):
        """切换温度调节模式"""
        if isinstance(new_mode, TempMode):
            self.mode = new_mode
            print(f"🔄 模式已切换为：{new_mode.value}")
            # 切换到关闭模式时停止风扇
            if new_mode == TempMode.OFF:
                self.fan_speed = 0
                self.is_running = False
                print("🔴 温度调节系统已关闭")
            else:
                self.is_running = True
                if self.fan_speed == 0:
                    self.fan_speed = 2  # 切换回运行模式时默认2档风速
        else:
            print("❌ 模式设置失败！请传入合法的TempMode枚举值")

    def set_fan_speed(self, speed):
        """设置风扇转速（1-5档）"""
        if 1 <= speed <= 5:
            self.fan_speed = speed
            print(f"🌬️  风扇转速已设置为：{speed}档")
        else:
            print("❌ 风速设置失败！请设置1~5档范围内的转速")

    def adjust_temp(self):
        """核心温度调节逻辑"""
        if not self.is_running:
            return

        current_temp = self.simulate_temp_sensor()
        target_temp = self.target_temp
        temp_diff = current_temp - target_temp

        # 根据模式执行调节逻辑
        if self.mode == TempMode.AUTO:
            # 自动模式：温差超过容差时触发制冷/制热
            if temp_diff > self.temp_tolerance:
                self._cooling()
            elif temp_diff < -self.temp_tolerance:
                self._heating()
            else:
                self._fan_only()  # 温度达标仅吹风

        elif self.mode == TempMode.COOL:
            self._cooling() if temp_diff > self.temp_tolerance else self._fan_only()

        elif self.mode == TempMode.HEAT:
            self._heating() if temp_diff < -self.temp_tolerance else self._fan_only()

        elif self.mode == TempMode.FAN:
            self._fan_only()

        # 打印当前状态
        self._print_status()

    def _cooling(self):
        """制冷逻辑：降低当前温度"""
        # 制冷效率与风扇转速正相关
        cool_rate = 0.2 * self.fan_speed
        self.current_temp -= cool_rate
        self.current_temp = max(self.min_temp, self.current_temp)  # 不低于最低温

    def _heating(self):
        """制热逻辑：升高当前温度"""
        heat_rate = 0.15 * self.fan_speed
        self.current_temp += heat_rate
        self.current_temp = min(self.max_temp, self.current_temp)  # 不高于最高温

    def _fan_only(self):
        """仅吹风：温度不变，维持通风"""
        pass

    def _print_status(self):
        """打印当前系统状态"""
        print(f"\n📊 当前系统状态：")
        print(f"  当前温度：{round(self.current_temp, 1)}℃ | 目标温度：{self.target_temp}℃")
        print(f"  运行模式：{self.mode.value} | 风扇转速：{self.fan_speed}档")
        print("-" * 40)

    def run(self, duration=10):
        """运行系统（模拟duration秒的调节过程）"""
        print("🚗 无人车温度调节系统启动...")
        start_time = time.time()
        while time.time() - start_time < duration:
            self.adjust_temp()
            time.sleep(1)  # 每秒调节一次
        print("⏹️  系统模拟运行结束")


# 测试示例
if __name__ == "__main__":
    # 初始化温度调节系统
    temp_system = AutoCarTempSystem()

    # 模拟场景1：初始温度25℃，设置目标22℃，自动模式运行5秒
    temp_system.set_target_temp(22.0)
    temp_system.run(duration=5)

    # 模拟场景2：切换到制热模式，设置目标28℃，风速4档，运行5秒
    temp_system.set_mode(TempMode.HEAT)
    temp_system.set_target_temp(28.0)
    temp_system.set_fan_speed(4)
    temp_system.run(duration=5)

    # 模拟场景3：切换到关闭模式
    temp_system.set_mode(TempMode.OFF)