import sys

sys.path.append(r"D:\Others\AirSim-1.8.1-windows\AirSim-1.8.1-windows\PythonClient")

import airsim
import time

print("尝试连接到 AirSim...")

try:
    # 连接到 AirSim 服务器
    client = airsim.CarClient()
    client.confirmConnection()
    print("✓ 成功连接到 AirSim 服务器！")

    # 获取车辆状态
    car_state = client.getCarState()
    print(f"车辆位置: {car_state.kinematics_estimated.position}")

except Exception as e:
    print(f"连接失败: {e}")
    print("请确保 AirSim 仿真环境正在运行")