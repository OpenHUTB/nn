import time
import random  # 仅用于模拟硬件数据，实际场景删除


class UnmannedVehicleBattery:
    """无人车电池电量管理类"""

    def __init__(self):
        # 电池参数配置（根据实际电池规格调整）
        self.max_voltage = 12.6  # 满电电压（12V锂电池为例）
        self.min_voltage = 10.0  # 欠压保护电压
        self.current_voltage = 0.0  # 当前电压
        self.battery_percent = 0.0  # 剩余电量百分比

    def read_battery_voltage(self):
        """
        读取电池电压（模拟硬件采集）
        实际场景：替换为ADC读取/串口接收BMS数据/I2C通信等
        """
        # 模拟电压波动（范围：10.0~12.6V）
        self.current_voltage = round(random.uniform(10.0, 12.6), 2)
        # 实际硬件示例（以树莓派ADC为例）：
        # import adafruit_ads1x15.ads1115 as ADS
        # from adafruit_ads1x15.analog_in import AnalogIn
        # i2c = board.I2C()
        # ads = ADS.ADS1115(i2c)
        # chan = AnalogIn(ads, ADS.P0)
        # self.current_voltage = chan.voltage * voltage_divider_ratio  # 电压分压比

    def calculate_battery_percent(self):
        """计算剩余电量百分比"""
        if self.current_voltage >= self.max_voltage:
            self.battery_percent = 100.0
        elif self.current_voltage <= self.min_voltage:
            self.battery_percent = 0.0
        else:
            # 线性计算（实际可根据电池放电曲线优化）
            self.battery_percent = round(
                (self.current_voltage - self.min_voltage) /
                (self.max_voltage - self.min_voltage) * 100,
                1
            )

    def get_battery_status(self):
        """判断电量状态"""
        if self.battery_percent >= 95:
            return "满电", "🟢"
        elif 20 <= self.battery_percent < 95:
            return "正常", "🟢"
        elif 5 <= self.battery_percent < 20:
            return "低电量", "🟡"
        else:
            return "紧急（请充电）", "🔴"

    def display_battery_info(self):
        """可视化显示电量信息"""
        # 清空控制台（可选）
        # os.system('cls' if os.name == 'nt' else 'clear')

        # 电量条可视化
        bar_length = 20
        filled_length = int(bar_length * self.battery_percent // 100)
        battery_bar = "█" * filled_length + "-" * (bar_length - filled_length)

        # 获取状态
        status, color = self.get_battery_status()

        # 打印信息
        print(f"\n=== 无人车电池状态 ===")
        print(f"当前电压: {self.current_voltage}V")
        print(f"剩余电量: |{battery_bar}| {self.battery_percent}%")
        print(f"状态: {color} {status}")

        # 低电量告警
        if self.battery_percent < 5:
            print("⚠️  电量过低，立即停止作业并充电")


def main():
    """主循环"""
    battery = UnmannedVehicleBattery()
    print("无人车电量监控系统启动...")

    try:
        while True:
            battery.read_battery_voltage()  # 读取电压
            battery.calculate_battery_percent()  # 计算电量
            battery.display_battery_info()  # 显示信息
            time.sleep(1)  # 1秒刷新一次
    except KeyboardInterrupt:
        print("\n监控系统已退出")


if __name__ == "__main__":
    main()