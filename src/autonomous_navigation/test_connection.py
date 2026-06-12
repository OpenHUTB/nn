import airsim
import time

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("连接成功！当前无人机状态：", client.getMultirotorState())

# 简单起飞和降落测试（可选）
client.takeoffAsync().join()
time.sleep(2)
client.landAsync().join()