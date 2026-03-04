# check_status.py
import airsim
import time

print("=" * 50)
print("无人机状态检查")
print("=" * 50)

# 连接到模拟器
client = airsim.MultirotorClient()
client.confirmConnection()
print("✓ 已连接到AbandonedPark模拟器")

# 检查无人机状态
state = client.getMultirotorState()
print(f"当前位置: X={state.kinematics_estimated.position.x_val:.1f}, "
      f"Y={state.kinematics_estimated.position.y_val:.1f}, "
      f"Z={state.kinematics_estimated.position.z_val:.1f}")
print(f"当前速度: {state.speed} m/s")
print(f"电池电量: {state.battery}%")
print(f"是否碰撞: {state.collision.has_collided}")

# 解锁无人机
print("\n解锁无人机...")
client.enableApiControl(True)
client.armDisarm(True)
print("✓ 无人机已解锁，准备就绪")

print("\n" + "=" * 50)
print("状态检查完成！无人机可以正常控制")
print("=" * 50)# check_status.py
import airsim
import time

print("=" * 50)
print("无人机状态检查")
print("=" * 50)

# 连接到模拟器
client = airsim.MultirotorClient()
client.confirmConnection()
print("✓ 已连接到AbandonedPark模拟器")

# 检查无人机状态
state = client.getMultirotorState()
print(f"当前位置: X={state.kinematics_estimated.position.x_val:.1f}, "
      f"Y={state.kinematics_estimated.position.y_val:.1f}, "
      f"Z={state.kinematics_estimated.position.z_val:.1f}")
print(f"当前速度: {state.speed} m/s")
print(f"电池电量: {state.battery}%")
print(f"是否碰撞: {state.collision.has_collided}")

# 解锁无人机
print("\n解锁无人机...")
client.enableApiControl(True)
client.armDisarm(True)
print("✓ 无人机已解锁，准备就绪")

print("\n" + "=" * 50)
print("状态检查完成！无人机可以正常控制")
print("=" * 50)