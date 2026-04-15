import airsim
import numpy as np
import keyboard
import time

# --- 配置 ---
VEHICLE_NAME = "Drone_1"
LIDAR_NAME = "lidar_1"

# 速度设置
H_SPEED = 3.0  # 水平移动速度 (m/s)
V_SPEED = 2.0  # 垂直升降速度 (m/s)
YAW_SPEED = 40.0  # 旋转速度 (度/秒)
MIN_DIST = 3.5  # 避障距离


def print_red(text): print(f"\033[91m{text}\033[0m")


# --- 连接与起飞 ---
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, vehicle_name=VEHICLE_NAME)
client.armDisarm(True, vehicle_name=VEHICLE_NAME)
client.takeoffAsync(vehicle_name=VEHICLE_NAME).join()
client.moveToPositionAsync(0, 0, -2, 3, vehicle_name=VEHICLE_NAME).join()

print("\n=== 避障系统启动 ===")
print("  控制键位:")
print("  [W/S] 前进/后退")
print("  [A/D] 向左/向右 (平移)")
print("  [Q/E] 左转/右转 (旋转机头) ")
print("  [↑/↓] 上升/下降")
print("  [Space] 悬停")


def analyze_lidar(client):
    """雷达避障分析 (保持之前的逻辑)"""
    lidar_data = client.getLidarData(lidar_name=LIDAR_NAME, vehicle_name=VEHICLE_NAME)
    blocked = {'front': False, 'back': False, 'left': False, 'right': False}
    front_dist = 999.0

    if not lidar_data or len(lidar_data.point_cloud) < 3:
        return blocked, front_dist

    points = np.array(lidar_data.point_cloud, dtype=np.float32)
    points = np.reshape(points, (int(points.shape[0] / 3), 3))

    # Z轴过滤 (保留上下1.5m范围)
    z_mask = (points[:, 2] > -1.5) & (points[:, 2] < 1.5)
    valid_points = points[z_mask]

    if len(valid_points) == 0: return blocked, front_dist

    # 计算正前方距离
    front_corridor_mask = (valid_points[:, 0] > 0) & (np.abs(valid_points[:, 1]) < 1.0)
    front_objs = valid_points[front_corridor_mask]
    if len(front_objs) > 0:
        front_dist = np.min(front_objs[:, 0])

    # 避障判定
    dist_sq = valid_points[:, 0] ** 2 + valid_points[:, 1] ** 2
    danger_mask = dist_sq < (MIN_DIST ** 2)
    danger_points = valid_points[danger_mask]

    width_threshold = 2.0
    for p in danger_points:
        x, y = p[0], p[1]
        if x > 0.5 and abs(y) < width_threshold:
            blocked['front'] = True
        elif x < -0.5 and abs(y) < width_threshold:
            blocked['back'] = True
        elif y < -0.5 and abs(x) < width_threshold:
            blocked['left'] = True
        elif y > 0.5 and abs(x) < width_threshold:
            blocked['right'] = True

    return blocked, front_dist


try:
    last_print = time.time()
    while True:
        # 1. 获取避障状态
        is_blocked, front_wall_dist = analyze_lidar(client)

        # 2. 打印状态
        if time.time() - last_print > 0.2:
            dist_str = f"{front_wall_dist:.2f}m" if front_wall_dist < 999 else "安全"
            print(f"\r[雷达] 前方距离: {dist_str} | 阻挡: {'🛑' if is_blocked['front'] else '✅'}      ", end="",
                  flush=True)
            last_print = time.time()

        # 3. 初始化速度
        vx, vy, vz = 0.0, 0.0, 0.0
        yaw_rate = 0.0  # 初始化旋转速度

        # 4. 读取键盘输入
        # --- 移动 ---
        if keyboard.is_pressed('w'): vx = H_SPEED
        if keyboard.is_pressed('s'): vx = -H_SPEED
        if keyboard.is_pressed('a'): vy = -H_SPEED
        if keyboard.is_pressed('d'): vy = H_SPEED

        # --- 升降 ---
        if keyboard.is_pressed('up'): vz = -V_SPEED
        if keyboard.is_pressed('down'): vz = V_SPEED

        # --- 旋转 (新增逻辑) ---
        if keyboard.is_pressed('q'): yaw_rate = -YAW_SPEED  # 左转 (逆时针)
        if keyboard.is_pressed('e'): yaw_rate = YAW_SPEED  # 右转 (顺时针)

        # --- 刹车 ---
        if keyboard.is_pressed('space'):
            vx, vy, vz, yaw_rate = 0.0, 0.0, 0.0, 0.0

        if keyboard.is_pressed('esc'): break

        # 5. 避障介入 (强制刹车)
        intervention = False
        if vx > 0 and is_blocked['front']: vx = 0.0; intervention = True
        if vx < 0 and is_blocked['back']: vx = 0.0; intervention = True
        if vy < 0 and is_blocked['left']: vy = 0.0; intervention = True
        if vy > 0 and is_blocked['right']: vy = 0.0; intervention = True

        if intervention:
            print(f"\n\033[91m🛑 [避障] 强制刹车! 距离: {front_wall_dist:.2f}m\033[0m")
            last_print = time.time()

        # 6. 发送指令 (关键修改)
        client.moveByVelocityAsync(
            vx=float(vx),
            vy=float(vy),
            vz=float(vz),
            duration=0.1,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            # --- 这里修改了 YawMode ---
            # is_rate=True 表示我们要控制旋转的“速度”
            # yaw_or_rate=yaw_rate 就是我们设置的度数/秒
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=float(yaw_rate)),
            vehicle_name=VEHICLE_NAME
        ).join()

except KeyboardInterrupt:
    pass
finally:
    print("\n降落...")
    client.landAsync(vehicle_name=VEHICLE_NAME).join()
    client.armDisarm(False, vehicle_name=VEHICLE_NAME)
    client.enableApiControl(False, vehicle_name=VEHICLE_NAME)