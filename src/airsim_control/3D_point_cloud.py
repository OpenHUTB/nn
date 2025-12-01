import airsim
import numpy as np
import keyboard
import time
import os

# --- 配置 ---
VEHICLE_NAME = "Drone_1"
LIDAR_NAME = "lidar_1"
H_SPEED = 2.0  # 水平移动速度
V_SPEED = 1.0  # 垂直移动速度
YAW_SPEED = 30.0  # 旋转速度

# ---设置绝对路径---
OUTPUT_FILE = r"D:\Others\map_output.asc"

# --- 检查目录是否存在 ---
output_dir = os.path.dirname(OUTPUT_FILE)
if not os.path.exists(output_dir):
    print(f"错误: 找不到文件夹 '{output_dir}'")
    print("请先手动创建这个文件夹，或者修改代码中的保存路径。")
    exit()


#数学工具：四元数转旋转矩阵
def get_rotation_matrix(q):
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]
    ])


# 初始化
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, vehicle_name=VEHICLE_NAME)
client.armDisarm(True, vehicle_name=VEHICLE_NAME)
client.takeoffAsync(vehicle_name=VEHICLE_NAME).join()
client.moveToPositionAsync(0, 0, -2, 3, vehicle_name=VEHICLE_NAME).join()

print("=== 3D 扫描模式启动 ===")
print("🎮 控制键位: [WASD]移动  [QE]旋转  [↑↓]升降")
print(f"📁 数据将保存到: {OUTPUT_FILE}")

# 清空旧文件
with open(OUTPUT_FILE, "w") as f:
    f.write("")

try:
    total_points_captured = 0
    last_save_time = time.time()
    points_buffer = []

    while True:
        # 1. 获取位姿
        state = client.simGetVehiclePose(vehicle_name=VEHICLE_NAME)
        pos = state.position
        orientation = state.orientation

        # 2. 获取雷达数据
        lidar_data = client.getLidarData(lidar_name=LIDAR_NAME, vehicle_name=VEHICLE_NAME)

        if lidar_data and len(lidar_data.point_cloud) >= 3:
            raw_points = np.array(lidar_data.point_cloud, dtype=np.float32)
            local_points = np.reshape(raw_points, (int(raw_points.shape[0] / 3), 3))

            # --- 坐标转换 ---
            R = get_rotation_matrix(orientation)
            rotated_points = np.dot(local_points, R.T)
            t_vec = np.array([pos.x_val, pos.y_val, pos.z_val])
            global_points = rotated_points + t_vec

            points_buffer.extend(global_points)
            total_points_captured += len(global_points)

        # 3. 写入文件
        if time.time() - last_save_time > 0.5:
            if points_buffer:
                with open(OUTPUT_FILE, "a") as f:
                    for p in points_buffer:
                        f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")

                print(f"\r[扫描中] 已采集点数: {total_points_captured} | 写入 D:\\Others...", end="")
                points_buffer = []
                last_save_time = time.time()

        # 4. 飞行控制
        vx, vy, vz = 0.0, 0.0, 0.0
        yaw_rate = 0.0

        if keyboard.is_pressed('w'): vx = H_SPEED
        if keyboard.is_pressed('s'): vx = -H_SPEED
        if keyboard.is_pressed('a'): vy = -H_SPEED
        if keyboard.is_pressed('d'): vy = H_SPEED

        if keyboard.is_pressed('up'): vz = -V_SPEED
        if keyboard.is_pressed('down'): vz = V_SPEED

        if keyboard.is_pressed('q'): yaw_rate = -YAW_SPEED
        if keyboard.is_pressed('e'): yaw_rate = YAW_SPEED

        if keyboard.is_pressed('esc'): break

        client.moveByVelocityAsync(
            vx, vy, vz, 0.1,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=float(yaw_rate)),
            vehicle_name=VEHICLE_NAME
        ).join()

except KeyboardInterrupt:
    pass
finally:
    if points_buffer:
        with open(OUTPUT_FILE, "a") as f:
            for p in points_buffer:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")

    print(f"\n扫描结束！文件已保存至: {OUTPUT_FILE}")
    client.reset()