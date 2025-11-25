import airsim
import numpy as np
import keyboard
import time

# --- 配置 ---
VEHICLE_NAME = "Drone_1"
LIDAR_NAME = "lidar_1"
H_SPEED = 3.0
V_SPEED = 2.0
MIN_DIST = 3.5  # 避障距离


def print_red(text): print(f"\033[91m{text}\033[0m")


# --- 连接 ---
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, vehicle_name=VEHICLE_NAME)
client.armDisarm(True, vehicle_name=VEHICLE_NAME)
client.takeoffAsync(vehicle_name=VEHICLE_NAME).join()
client.moveToPositionAsync(0, 0, -2, 3, vehicle_name=VEHICLE_NAME).join()

print("\n=== 避障系统启动 ===")

def analyze_lidar(client):
    """
    分析雷达数据
    返回:
    1. blocked (方向封锁状态 dict)
    2. front_dist (正前方墙壁的精确距离，如果没有则为 999)
    """
    lidar_data = client.getLidarData(lidar_name=LIDAR_NAME, vehicle_name=VEHICLE_NAME)
    blocked = {'front': False, 'back': False, 'left': False, 'right': False}
    front_dist = 999.0

    if not lidar_data or len(lidar_data.point_cloud) < 3:
        return blocked, front_dist

    points = np.array(lidar_data.point_cloud, dtype=np.float32)
    points = np.reshape(points, (int(points.shape[0] / 3), 3))

    # --- 1. 核心过滤：只看无人机高度范围内的点 ---
    # Z轴过滤：保留上下 1.5米 范围内的点 (过滤掉地面)
    z_mask = (points[:, 2] > -1.5) & (points[:, 2] < 1.5)
    valid_points = points[z_mask]

    if len(valid_points) == 0:
        return blocked, front_dist

    # --- 2. 计算正前方距离 (调试用) ---
    # 定义正前方：X > 0 且 |Y| < 1.0 (只看正中间 2米宽的走廊)
    front_corridor_mask = (valid_points[:, 0] > 0) & (np.abs(valid_points[:, 1]) < 1.0)
    front_objs = valid_points[front_corridor_mask]

    if len(front_objs) > 0:
        # 在正前方走廊里，找 X 最小的值
        front_dist = np.min(front_objs[:, 0])

    # --- 3. 避障判定 (控制用) ---
    # 使用平面距离平方来判定
    dist_sq = valid_points[:, 0] ** 2 + valid_points[:, 1] ** 2
    danger_mask = dist_sq < (MIN_DIST ** 2)
    danger_points = valid_points[danger_mask]

    # 判定方位
    width_threshold = 2.0  # 判定宽度
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
        # 获取分析结果
        is_blocked, front_wall_dist = analyze_lidar(client)

        # 实时打印前方距离 (每0.2秒刷新一次)
        if time.time() - last_print > 0.2:
            dist_str = f"{front_wall_dist:.2f}m" if front_wall_dist < 999 else "安全"
            # 这里的 \r 保证在同一行刷新
            print(
                f"\r[雷达监测] 正前方墙壁距离: {dist_str}  |  状态: {'🛑阻挡' if is_blocked['front'] else '✅通行'}      ",
                end="", flush=True)
            last_print = time.time()

        # 读取键盘
        vx, vy, vz = 0.0, 0.0, 0.0
        if keyboard.is_pressed('w'): vx = H_SPEED
        if keyboard.is_pressed('s'): vx = -H_SPEED
        if keyboard.is_pressed('a'): vy = -H_SPEED
        if keyboard.is_pressed('d'): vy = H_SPEED
        if keyboard.is_pressed('up'): vz = -V_SPEED
        if keyboard.is_pressed('down'): vz = V_SPEED
        if keyboard.is_pressed('space'): vx, vy, vz = 0.0, 0.0, 0.0
        if keyboard.is_pressed('esc'): break

        # 避障介入
        intervention = False
        if vx > 0 and is_blocked['front']: vx = 0.0; intervention = True
        if vx < 0 and is_blocked['back']: vx = 0.0; intervention = True
        if vy < 0 and is_blocked['left']: vy = 0.0; intervention = True
        if vy > 0 and is_blocked['right']: vy = 0.0; intervention = True

        if intervention:
            # 如果触发避障，强制刷新一行红字，防止被 \r 覆盖看不清
            print(f"\n\033[91m🛑 [避障系统] 强制刹车! 前方距离: {front_wall_dist:.2f}m\033[0m")
            last_print = time.time()  # 重置打印时间

        # 执行指令
        client.moveByVelocityAsync(
            vx=float(vx), vy=float(vy), vz=float(vz), duration=0.1,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
            vehicle_name=VEHICLE_NAME
        ).join()

except KeyboardInterrupt:
    pass
finally:
    print("\n降落...")
    client.landAsync(vehicle_name=VEHICLE_NAME).join()
    client.armDisarm(False, vehicle_name=VEHICLE_NAME)
    client.enableApiControl(False, vehicle_name=VEHICLE_NAME)