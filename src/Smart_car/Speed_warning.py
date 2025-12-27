import time
import serial  # 需安装：pip install pyserial
import random  # 仿真速度数据（真实场景替换为传感器读取）

# ------------------- 配置参数 -------------------
# 限速阈值（km/h）
SPEED_LIMIT = 20
# 预警等级配置
WARNING_LEVELS = {
    "low": (SPEED_LIMIT, SPEED_LIMIT + 5),  # 轻度超速：20-25km/h
    "medium": (SPEED_LIMIT + 5, SPEED_LIMIT + 10),  # 中度超速：25-30km/h
    "high": (SPEED_LIMIT + 10, float('inf'))  # 重度超速：>30km/h
}
# 串口配置（对接车载显示屏/报警器，真实场景启用）
SERIAL_PORT = "COM3"  # Windows: COM3 / Linux: /dev/ttyUSB0
BAUD_RATE = 9600
ser = None


# ------------------- 初始化函数 -------------------
def init_serial():
    """初始化串口（用于向硬件发送预警指令）"""
    global ser
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Serial port {SERIAL_PORT} initialized successfully")
    except Exception as e:
        print(f"Serial port init failed: {e}")
        ser = None


# ------------------- 速度读取函数 -------------------
def read_vehicle_speed():
    """
    读取车辆速度（真实场景替换为传感器/总线数据）
    返回：当前速度（km/h）
    """
    # 仿真：随机生成10-35km/h的速度（真实场景删除此段）
    simulated_speed = random.uniform(10, 35)
    return round(simulated_speed, 1)

    # 真实场景示例：从串口读取速度传感器数据
    # if ser and ser.in_waiting > 0:
    #     speed_data = ser.readline().decode('utf-8').strip()
    #     return float(speed_data) if speed_data else 0.0
    # return 0.0


# ------------------- 超速预警核心函数 -------------------
def speed_warning(current_speed):
    """
    超速预警判断与输出
    :param current_speed: 当前速度（km/h）
    :return: 预警等级（None/low/medium/high）、预警信息（英文）
    """
    if current_speed < SPEED_LIMIT:
        return None, f"Current speed: {current_speed} km/h - Normal"

    # 判断预警等级
    warning_level = None
    warning_msg = ""
    if WARNING_LEVELS["low"][0] <= current_speed < WARNING_LEVELS["low"][1]:
        warning_level = "low"
        warning_msg = f"Speed Warning: {current_speed} km/h (Over limit by {current_speed - SPEED_LIMIT} km/h) - Slow down!"
    elif WARNING_LEVELS["medium"][0] <= current_speed < WARNING_LEVELS["medium"][1]:
        warning_level = "medium"
        warning_msg = f"Over Speed Warning: {current_speed} km/h - Reduce speed immediately!"
    elif current_speed >= WARNING_LEVELS["high"][0]:
        warning_level = "high"
        warning_msg = f"CRITICAL Over Speed Warning: {current_speed} km/h - STOP VEHICLE!"

    # 输出预警（控制台 + 串口/硬件）
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {warning_msg}")
    if ser:
        ser.write(f"{warning_level}:{warning_msg}\n".encode('utf-8'))

    return warning_level, warning_msg


# ------------------- 主循环 -------------------
def main():
    # 初始化串口（可选）
    # init_serial()

    print(f"Autonomous Vehicle Speed Warning System Started | Speed Limit: {SPEED_LIMIT} km/h")
    try:
        while True:
            # 读取当前速度
            current_speed = read_vehicle_speed()
            # 触发超速预警
            speed_warning(current_speed)
            # 1秒刷新一次（可根据传感器频率调整）
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nSystem stopped by user")
    finally:
        if ser:
            ser.close()


if __name__ == "__main__":
    main()