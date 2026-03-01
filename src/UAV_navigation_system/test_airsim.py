import airsim
import sys

print("Python 版本:", sys.version)
print("AirSim 版本:", airsim.__version__)
print("AirSim 导入成功！")

# 检查客户端是否可以创建
client = airsim.MultirotorClient()
print("客户端创建成功")