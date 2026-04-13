import airsim
import time

# 连接无人机
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("✅ 成功连接 AirSim 无人机！")

# 起飞
client.takeoffAsync().join()
print("🚁 无人机已起飞！")
time.sleep(2)

# 自动巡航
print("🔁 启动自动巡航...")
client.moveToPositionAsync(5, 0, -5, 5).join()
time.sleep(1)
client.moveToPositionAsync(5, 5, -5, 5).join()
time.sleep(1)
client.moveToPositionAsync(0, 5, -5, 5).join()
time.sleep(1)
client.moveToPositionAsync(0, 0, -5, 5).join()

# 降落
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)

print("🏁 任务完成！无人机已降落！")
