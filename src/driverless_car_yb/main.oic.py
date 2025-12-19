import time
import logging
from datetime import datetime
import threading

# -------------------------- 配置参数 --------------------------
# 电池参数（根据实际电池型号调整）
BATTERY_FULL_VOLTAGE = 12.6  # 满电电压（12V锂电池为例）
BATTERY_EMPTY_VOLTAGE = 10.5  # 亏电电压（保护电压）
LOW_BATTERY_THRESHOLD = 20  # 低电量报警阈值（百分比）
CRITICAL_BATTERY_THRESHOLD = 10  # 紧急低电量阈值（百分比）

# 日志配置
LOG_FILE = "battery_log.txt"
LOG_LEVEL = logging.INFO

# 监测频率（秒/次）
MONITOR_INTERVAL = 1

# 硬件适配标记（True=使用真实硬件ADC，False=模拟数据）
USE_HARDWARE_ADC = False
# ---------------------------------------------------------------

# 初始化日志
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)


class UnmannedVehicleBatteryMonitor:
    def __init__(self):
        self.current_voltage = 0.0  # 当前电池电压
        self.current_soc = 0  # 剩余电量百分比（State of Charge）
        self.estimated_range = 0.0  # 预估续航里程（km）
        self.is_low_battery = False  # 低电量状态
        self.is_critical_battery = False  # 紧急低电量状态
        self.running = True  # 监测线程运行标记

    def _read_battery_voltage(self) -> float:
        """
        读取电池电压（核心硬件接口）
        实际场景需根据硬件调整：
        - 嵌入式系统（RPi）：使用ADC引脚（如ADS1115模块）
        - 串口设备：通过CAN总线/串口读取BMS数据
        - 模拟模式：生成随机电压用于测试
        """
        if USE_HARDWARE_ADC:
            # -------------------------- 硬件ADC读取示例（Raspberry Pi + ADS1115）--------------------------
            # 需安装依赖：pip install adafruit-circuitpython-ads1x15
            try:
                import board
                import busio
                import adafruit_ads1x15.ads1115 as ADS
                from adafruit_ads1x15.analog_in import AnalogIn

                # 初始化I2C总线
                i2c = busio.I2C(board.SCL, board.SDA)
                ads = ADS.ADS1115(i2c)
                chan = AnalogIn(ads, ADS.P0)  # 电池电压接入P0引脚

                # 电压分压计算（电池电压通常高于ADC量程，需串联电阻分压）
                # 例：分压比 = (R1 + R2)/R2，假设R1=100kΩ, R2=100kΩ → 分压比=2
                voltage_divider_ratio = 2.0
                raw_voltage = chan.voltage * voltage_divider_ratio
                return round(raw_voltage, 2)
            except Exception as e:
                logging.error(f"硬件电压读取失败：{str(e)}")
                return self.current_voltage  # 异常时返回上次值
        else:
            # 模拟电压：在满电和亏电之间随机波动（用于测试）
            import random
            voltage = random.uniform(BATTERY_EMPTY_VOLTAGE + 0.1, BATTERY_FULL_VOLTAGE)
            return round(voltage, 2)

    def _calculate_soc(self, voltage: float) -> int:
        """
        根据电压计算剩余电量百分比（SOC）
        线性计算（实际场景可替换为更精准的SOC算法，如库仑计法）
        """
        if voltage >= BATTERY_FULL_VOLTAGE:
            return 100
        elif voltage <= BATTERY_EMPTY_VOLTAGE:
            return 0
        else:
            soc = ((voltage - BATTERY_EMPTY_VOLTAGE) /
                   (BATTERY_FULL_VOLTAGE - BATTERY_EMPTY_VOLTAGE)) * 100
            return int(round(soc))

    def _estimate_range(self, soc: int) -> float:
        """
        根据剩余电量预估续航里程（简化模型）
        实际场景需结合：平均功耗、车速、路况等参数
        """
        full_range = 100.0  # 满电续航（km，根据实际车型调整）
        return round((soc / 100) * full_range, 1)

    def _check_battery_alarm(self, soc: int):
        """
        检查低电量报警状态
        """
        self.is_low_battery = soc <= LOW_BATTERY_THRESHOLD
        self.is_critical_battery = soc <= CRITICAL_BATTERY_THRESHOLD

        if self.is_critical_battery:
            logging.critical(f"紧急低电量！剩余电量：{soc}%，请立即充电！")
            self._trigger_alarm("critical")
        elif self.is_low_battery:
            logging.warning(f"低电量提醒！剩余电量：{soc}%，建议尽快充电")
            self._trigger_alarm("low")

    def _trigger_alarm(self, alarm_type: str):
        """
        触发报警（可对接硬件：蜂鸣器、LED、语音等）
        """
        if USE_HARDWARE_ADC:
            # -------------------------- 硬件报警示例（Raspberry Pi GPIO）--------------------------
            # 需安装依赖：pip install RPi.GPIO
            try:
                import RPi.GPIO as GPIO
                BUZZER_PIN = 18  # 蜂鸣器GPIO引脚
                LED_PIN = 23  # LED GPIO引脚

                GPIO.setmode(GPIO.BCM)
                GPIO.setup(BUZZER_PIN, GPIO.OUT)
                GPIO.setup(LED_PIN, GPIO.OUT)

                # 紧急报警：蜂鸣器长鸣 + LED快闪
                if alarm_type == "critical":
                    GPIO.output(BUZZER_PIN, GPIO.HIGH)
                    for _ in range(5):
                        GPIO.output(LED_PIN, GPIO.HIGH)
                        time.sleep(0.2)
                        GPIO.output(LED_PIN, GPIO.LOW)
                        time.sleep(0.2)
                    GPIO.output(BUZZER_PIN, GPIO.LOW)
                # 低电量报警：蜂鸣器短鸣 + LED慢闪
                elif alarm_type == "low":
                    GPIO.output(BUZZER_PIN, GPIO.HIGH)
                    time.sleep(0.5)
                    GPIO.output(BUZZER_PIN, GPIO.LOW)
                    GPIO.output(LED_PIN, GPIO.HIGH)
                    time.sleep(1)
                    GPIO.output(LED_PIN, GPIO.LOW)
            except Exception as e:
                logging.error(f"硬件报警触发失败：{str(e)}")
        else:
            # 模拟报警：终端输出提示
            if alarm_type == "critical":
                print("\n" + "=" * 50)
                print("⚠️  紧急低电量报警 ⚠️")
                print(f"剩余电量：{self.current_soc}%")
                print("请立即停止运行并充电！")
                print("=" * 50 + "\n")
            elif alarm_type == "low":
                print("\n" + "-" * 50)
                print("⚠️  低电量提醒 ⚠️")
                print(f"剩余电量：{self.current_soc}%")
                print("建议尽快充电！")
                print("-" * 50 + "\n")

    def _display_battery_info(self):
        """
        终端可视化显示电量信息
        """
        # 电量图标（根据SOC生成进度条）
        bar_length = 20
        filled_length = int((self.current_soc / 100) * bar_length)
        battery_bar = "█" * filled_length + "░" * (bar_length - filled_length)

        # 状态颜色标记（终端ANSI颜色）
        if self.is_critical_battery:
            color = "\033[91m"  # 红色
        elif self.is_low_battery:
            color = "\033[93m"  # 黄色
        else:
            color = "\033[92m"  # 绿色
        reset_color = "\033[0m"

        # 清空终端并显示（兼容Windows/Linux）
        import os
        os.system("cls" if os.name == "nt" else "clear")

        print(f"{'=' * 60}")
        print(f"{'无人车电池监测系统':^60}")
        print(f"{'=' * 60}")
        print(f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"电池电压：{self.current_voltage}V (满电：{BATTERY_FULL_VOLTAGE}V / 亏电：{BATTERY_EMPTY_VOLTAGE}V)")
        print(f"剩余电量：{color}{self.current_soc}%{reset_color} | [{battery_bar}]")
        print(f"预估续航：{self.estimated_range}km (满电续航：100km)")
        print(f"状态：{self._get_battery_status_text()}")
        print(f"{'=' * 60}")
        print("提示：按 Ctrl+C 退出监测")

    def _get_battery_status_text(self) -> str:
        """
        获取电池状态描述文本
        """
        if self.is_critical_battery:
            return "🔴 紧急低电量（禁止运行）"
        elif self.is_low_battery:
            return "🟡 低电量（建议充电）"
        elif self.current_soc >= 80:
            return "🟢 满电状态"
        else:
            return "🟢 正常状态"

    def monitor_loop(self):
        """
        主监测循环（后台线程运行）
        """
        logging.info("无人车电池监测系统启动成功！")
        while self.running:
            try:
                # 1. 读取电压
                self.current_voltage = self._read_battery_voltage()

                # 2. 计算SOC和续航
                self.current_soc = self._calculate_soc(self.current_voltage)
                self.estimated_range = self._estimate_range(self.current_soc)

                # 3. 检查报警
                self._check_battery_alarm(self.current_soc)

                # 4. 显示信息
                self._display_battery_info()

                # 5. 记录日志
                logging.info(
                    f"电压：{self.current_voltage}V | "
                    f"SOC：{self.current_soc}% | "
                    f"续航：{self.estimated_range}km | "
                    f"状态：{self._get_battery_status_text()}"
                )

                # 6. 延时等待
                time.sleep(MONITOR_INTERVAL)
            except KeyboardInterrupt:
                logging.info("用户主动退出监测系统")
                self.running = False
            except Exception as e:
                logging.error(f"监测循环异常：{str(e)}", exc_info=True)
                time.sleep(1)  # 异常后延时重试

    def start(self):
        """
        启动监测系统（支持后台线程运行）
        """
        monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        monitor_thread.start()

        # 主线程等待用户中断
        try:
            while self.running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.running = False
            monitor_thread.join()
            logging.info("监测系统已退出")


if __name__ == "__main__":
    # 初始化并启动监测系统
    battery_monitor = UnmannedVehicleBatteryMonitor()
    battery_monitor.start()