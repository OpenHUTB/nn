import airsim
import numpy as np
import keyboard
import time
import os

# --- 1. 配置参数 ---
VEHICLE_NAME = "Drone_1"
LIDAR_NAME = "lidar_1"
H_SPEED = 2.0
V_SPEED = 1.0
YAW_SPEED = 30.0

# 最小记录阈值 防止悬停时产生大量冗余数据
MIN_MOVE_DIST = 0.05  # 至少移动 5厘米 才记录
MIN_ROT_ANGLE = 1.0  # 至少旋转 1度 才记录

OUTPUT_FILE = r"D:\Others\map_output_pro.asc"

# 检查目录
output_dir = os.path.dirname(OUTPUT_FILE)
if not os.path.exists(output_dir):
    print(f"错误: 找不到文件夹 '{output_dir}'")
    exit()


# --- 2. 数学工具 ---
def get_rotation_matrix(q):
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    # 归一化四元数 (防止数值漂移)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]
    ])


def calc_distance(p1, p2):
    return np.sqrt((p1.x_val - p2.x_val) ** 2 + (p1.y_val - p2.y_val) ** 2 + (p1.z_val - p2.z_val) ** 2)


# --- 3. 初始化 ---
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, vehicle_name=VEHICLE_NAME)
client.armDisarm(True, vehicle_name=VEHICLE_NAME)
client.takeoffAsync(vehicle_name=VEHICLE_NAME).join()
client.moveToPositionAsync(0, 0, -2, 3, vehicle_name=VEHICLE_NAME).join()

print("\n=== 3D 扫描系统 (专业版) ===")
print("控制: [WASD]移动 [QE]旋转 [↑↓]升降")
print("开关: [R] 键")
print("优化: 已启用坐标修正(ENU)与智能去重")

with open(OUTPUT_FILE, "w") as f:
    f.write("")

try:
    total_points = 0
    last_save_time = time.time()
    points_buffer = []
    is_recording = False

    # 用于智能去重的变量
    last_rec_pos = airsim.Vector3r(0, 0, 0)
    last_rec_yaw = 0.0

    while True:
        # --- 开关逻辑 ---
        if keyboard.is_pressed('r'):
            is_recording = not is_recording
            if is_recording:
                # 开启瞬间重置一下记录位置，确保能立刻记录
                last_rec_pos = airsim.Vector3r(999, 999, 999)
                print(f"\n>>> 录制开始...")
            else:
                print(f"\n>>> 录制暂停。")
            time.sleep(0.3)

        # --- 录制逻辑 ---
        if is_recording:
            # 1. 获取状态
            state = client.simGetVehiclePose(vehicle_name=VEHICLE_NAME)
            pos = state.position
            orientation = state.orientation

            # 计算当前偏航角 (简单估算)
            current_yaw = airsim.to_eularian_angles(orientation)[2]

            # 2. 智能去重判断 (核心优化)
            # 只有当位置移动超过阈值，或者旋转超过阈值时，才处理数据
            dist_moved = calc_distance(pos, last_rec_pos)
            rot_moved = abs(current_yaw - last_rec_yaw)

            if dist_moved > MIN_MOVE_DIST or rot_moved > np.deg2rad(MIN_ROT_ANGLE):

                # 更新上一次记录的位置
                last_rec_pos = pos
                last_rec_yaw = current_yaw

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

                    # --- 3. 坐标系修正 (NED -> ENU) ---
                    # AirSim Z是向下，我们把它取反，变成向上，这样在软件里看房子就是正的了
                    global_points[:, 2] = -global_points[:, 2]

                    # --- 4. 增加强度信息 (Intensity) ---
                    # 我们用高度(Z)作为第4列强度值，这样导入时可以直接按颜色显示
                    # 也可以用距离作为强度: intensity = np.linalg.norm(local_points, axis=1)
                    intensity = global_points[:, 2]

                    # 拼接数据: X Y Z I
                    data_to_save = np.column_stack((global_points, intensity))

                    points_buffer.extend(data_to_save)
                    total_points += len(global_points)

            # 写入硬盘
            if time.time() - last_save_time > 0.5:
                if points_buffer:
                    with open(OUTPUT_FILE, "a") as f:
                        for p in points_buffer:
                            # 保存4位小数，第4列是强度
                            f.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {p[3]:.3f}\n")

                    print(f"\r[REC] 点数: {total_points} | 状态: 移动扫描中...", end="")
                    points_buffer = []
                    last_save_time = time.time()
                else:
                    # 如果没有新数据(因为没移动)，打印静止状态
                    print(f"\r[REC] 点数: {total_points} | 状态: 等待移动...  ", end="")

        else:
            time.sleep(0.05)

        # --- 飞行控制 ---
        vx, vy, vz, yaw_rate = 0.0, 0.0, 0.0, 0.0
        if keyboard.is_pressed('w'): vx = H_SPEED
        if keyboard.is_pressed('s'): vx = -H_SPEED
        if keyboard.is_pressed('a'): vy = -H_SPEED
        if keyboard.is_pressed('d'): vy = H_SPEED
        if keyboard.is_pressed('up'): vz = -V_SPEED  # 注意：AirSim里负数是向上，但我们上面保存时已经反转了Z，这里控制依然遵循AirSim逻辑
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
                f.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {p[3]:.3f}\n")
    print(f"\n结束。文件: {OUTPUT_FILE}")
    client.reset()