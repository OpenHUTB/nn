import airsim
import time

# --- 与 settings.json 匹配 ---
VEHICLE_NAME = "Drone_1"
LIDAR_NAME = "lidar_1"
# -----------------------------------

client = airsim.MultirotorClient()
client.confirmConnection()
# API控制为 True 才能获取传感器数据
client.enableApiControl(True, vehicle_name=VEHICLE_NAME)

print(f"Connected. Testing Lidar '{LIDAR_NAME}' on vehicle '{VEHICLE_NAME}'...")
print("Please check the UE simulation window for RED debug points.")

try:
    for i in range(10):
        # 须同时指定 lidar_name 和 vehicle_name
        lidar_data = client.getLidarData(lidar_name=LIDAR_NAME, vehicle_name=VEHICLE_NAME)

        if lidar_data and lidar_data.point_cloud:
            num_points = len(lidar_data.point_cloud) // 3
            print(f"  > OK! Reading {i + 1}/10: Found {num_points} points.")

            # 打印第一个探测到的点
            # point_cloud 是一个扁平的 [x1, y1, z1, x2, y2, z2, ...] 列表
            first_point = lidar_data.point_cloud[0:3]
            print(f"    - First point (relative to Lidar): "
                  f"X={first_point[0]:.2f}, Y={first_point[1]:.2f}, Z={first_point[2]:.2f}")

        elif lidar_data and not lidar_data.point_cloud:
            print(f"  > Lidar '{LIDAR_NAME}' is working, but detected 0 points.")

        else:
            print(f"  > FAILED to get Lidar data. Check names and restart UE.")
            break  # 退出循环

        time.sleep(1.0)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    client.enableApiControl(False, vehicle_name=VEHICLE_NAME)
    print("Test complete.")