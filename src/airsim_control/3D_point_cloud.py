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

# 保存路径
OUTPUT_FILE = r"D:\Others\map_output.asc"

# 检查目录
output_dir = os.path.dirname(OUTPUT_FILE)
if not os.path.exists(output_dir):
    print(f"错误: 找不到文件夹 '{output_dir}'，请先创建。")
    exit()


# --- 数学工具 ---
def get_rotation_matrix(q):
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]
    ])


# --- 初始化 ---
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, vehicle_name=VEHICLE_NAME)
client.armDisarm(True, vehicle_name=VEHICLE_NAME)
client.takeoffAsync(vehicle_name=VEHICLE_NAME).join()
client.moveToPositionAsync(0, 0, -2, 3, vehicle_name=VEHICLE_NAME).join()

print("\n3D 扫描系统 手动控制版 ")
print("飞行控制: [WASD]移动  [QE]旋转  [↑↓]升降")
print("扫描开关: 按 [R] 键开启/停止录制")
print(f"文件路径: {OUTPUT_FILE}")

# 清空/初始化文件
with open(OUTPUT_FILE, "w") as f:
    f.write("")

try:
    total_points_captured = 0
    last_save_time = time.time()
    points_buffer = []

    # --- 新增：扫描状态标记 ---
    is_scanning = False

    while True:
        # --- 1. 监听开关按键 [R] ---
        if keyboard.is_pressed('r'):
            is_scanning = not is_scanning  # 切换状态
            if is_scanning:
                print(f"\n>>>开始录制数据... (当前总点数: {total_points_captured})")
            else:
                print(f"\n>>>暂停录制数据。")

            time.sleep(0.3)  # 简单的防抖动，防止按一下触发多次

        # --- 2. 仅在开启状态下处理数据 ---
        if is_scanning:
            # 获取位姿
            state = client.simGetVehiclePose(vehicle_name=VEHICLE_NAME)
            pos = state.position
            orientation = state.orientation

            # 获取雷达
            lidar_data = client.getLidarData(lidar_name=LIDAR_NAME, vehicle_name=VEHICLE_NAME)

            if lidar_data and len(lidar_data.point_cloud) >= 3:
                raw_points = np.array(lidar_data.point_cloud, dtype=np.float32)
                local_points = np.reshape(raw_points, (int(raw_points.shape[0] / 3), 3))

                # 坐标转换
                R = get_rotation_matrix(orientation)
                rotated_points = np.dot(local_points, R.T)
                t_vec = np.array([pos.x_val, pos.y_val, pos.z_val])
                global_points = rotated_points + t_vec

                points_buffer.extend(global_points)
                total_points_captured += len(global_points)

            # 写入文件 (每0.5秒)
            if time.time() - last_save_time > 0.5:
                if points_buffer:
                    with open(OUTPUT_FILE, "a") as f:
                        for p in points_buffer:
                            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")

                    # 使用 \r 动态刷新同一行，不刷屏
                    print(f"\r[录制中] 已采集: {total_points_captured} 点 | 正在写入...", end="")
                    points_buffer = []
                    last_save_time = time.time()
        else:
            # 暂停状态下，稍微sleep一下减少CPU占用，且不打印刷屏
            time.sleep(0.05)

        # --- 3. 飞行控制 (始终有效) ---
        vx, vy, vz, yaw_rate = 0.0, 0.0, 0.0, 0.0

        if keyboard.is_pressed('w'): vx = H_SPEED
        if keyboard.is_pressed('s'): vx = -H_SPEED
        if keyboard.is_pressed('a'): vy = -H_SPEED
        if keyboard.is_pressed('d'): vy = H_SPEED

        if keyboard.is_pressed('up'): vz = -V_SPEED
        if keyboard.is_pressed('down'): vz = V_SPEED

        if keyboard.is_pressed('q'): yaw_rate = -YAW_SPEED
        if keyboard.is_pressed('e'): yaw_rate = YAW_SPEED

        if keyboard.is_pressed('esc'): break

        # 发送控制指令
        client.moveByVelocityAsync(
            vx, vy, vz, 0.1,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=float(yaw_rate)),
            vehicle_name=VEHICLE_NAME
        ).join()

except KeyboardInterrupt:
    pass
finally:
    # 保存最后残留的数据
    if points_buffer:
        with open(OUTPUT_FILE, "a") as f:
            for p in points_buffer:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")

    print(f"\n\n任务结束！最终点数: {total_points_captured}")
    print(f"结果已保存: {OUTPUT_FILE}")
    client.reset()