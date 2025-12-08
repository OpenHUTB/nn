import airsim
import numpy as np
import time
import math

# --- 配置 ---
VEHICLE_NAME = "Drone_1"
LIDAR_NAME = "lidar_1"

# 飞行参数
TARGET_HEIGHT = -1.5
CRUISE_SPEED = 1.5
TURN_SPEED = 40.0
STOP_DIST = 4.0
EMERGENCY_DIST = 1.0

# --- 初始化 ---
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, vehicle_name=VEHICLE_NAME)
client.armDisarm(True, vehicle_name=VEHICLE_NAME)

print("🚀 起飞中...")
client.takeoffAsync(vehicle_name=VEHICLE_NAME).join()
client.moveToZAsync(TARGET_HEIGHT, 1, vehicle_name=VEHICLE_NAME).join()

print(f"\n=== 最终版: 机身坐标系飞行 (Body Frame) ===")


def get_front_distance():
    """获取正前方的障碍物距离"""
    lidar_data = client.getLidarData(lidar_name=LIDAR_NAME, vehicle_name=VEHICLE_NAME)
    if not lidar_data or len(lidar_data.point_cloud) < 3: return 99.0

    points = np.array(lidar_data.point_cloud, dtype=np.float32)
    points = np.reshape(points, (int(points.shape[0] / 3), 3))

    # 这里的过滤逻辑不需要变，因为 Lidar 数据本身就是相对于机身的(Body Frame)
    valid_points = points[(points[:, 2] > -0.5) & (points[:, 2] < 0.5)]
    front_mask = (valid_points[:, 0] > 0) & (np.abs(valid_points[:, 1]) < 0.8)
    front_objs = valid_points[front_mask]

    if len(front_objs) > 0:
        return np.min(front_objs[:, 0])
    return 99.0


def turn_by_time(angle):
    """盲转"""
    direction_str = "右" if angle > 0 else "左"
    print(f"   ↪️ 正在向{direction_str}转 {abs(angle)}° ...")

    duration = abs(angle) / TURN_SPEED
    yaw_rate = TURN_SPEED if angle > 0 else -TURN_SPEED

    # 旋转时速度设为0，原地转
    client.moveByVelocityAsync(0, 0, 0, duration,
                               drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                               yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=float(yaw_rate)),
                               vehicle_name=VEHICLE_NAME).join()

    client.moveByVelocityAsync(0, 0, 0, 0.5, vehicle_name=VEHICLE_NAME).join()


def emergency_brake():
    print("🚨 距离过近！强制反推刹车！")
    # BodyFrame 下，vx=-1 就是向后退，不用管此时机头朝哪
    client.moveByVelocityBodyFrameAsync(-1.0, 0, 0, 0.8,
                                        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                        yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
                                        vehicle_name=VEHICLE_NAME).join()

    client.moveByVelocityAsync(0, 0, 0, 0.5, vehicle_name=VEHICLE_NAME).join()


def decide_direction():
    """停车决策逻辑"""
    print("\n🛑 停车决策中...")

    # 1. 左转90度
    turn_by_time(-90)
    left_dist = get_front_distance()
    print(f"      👀 左侧视野: {left_dist:.1f}m")

    # 2. 右转180度 (看向右边)
    turn_by_time(180)
    right_dist = get_front_distance()
    print(f"      👀 右侧视野: {right_dist:.1f}m")

    # 3. 决策
    if left_dist < 3.0 and right_dist < 3.0:
        print("⚠️ 死胡同 -> 继续右转90度 (掉头)")
        turn_by_time(90)

    elif left_dist > right_dist:
        print("✅ 左边宽敞 -> 左转180度")
        turn_by_time(-180)
    else:
        print("✅ 右边宽敞 -> 保持当前方向")
        pass
    return


try:
    while True:
        front_dist = get_front_distance()

        # --- 1. 紧急避险 ---
        if front_dist < EMERGENCY_DIST:
            emergency_brake()
            decide_direction()
            continue

        # --- 2. 遇阻停车 ---
        if front_dist < STOP_DIST:
            print(f"\r[🛑 刹车] 前方障碍 {front_dist:.1f}m < {STOP_DIST}m   ", end="", flush=True)
            client.moveByVelocityAsync(0, 0, 0, 0.5, vehicle_name=VEHICLE_NAME).join()

            if get_front_distance() < STOP_DIST:
                decide_direction()

        # --- 3. 正常巡航 ---
        else:
            print(f"\r[🚀 巡航] 前方: {front_dist:.1f}m   ", end="", flush=True)

            # 高度控制 (依然是 Global Z)
            z_current = client.simGetVehiclePose(vehicle_name=VEHICLE_NAME).position.z_val
            vz = (TARGET_HEIGHT - z_current) * 1.0

            # ---------------------------------------------------------
            # 🔴 关键修复：使用 moveByVelocityBodyFrameAsync
            # vx = CRUISE_SPEED (正数) 现在代表 "机头正前方"
            # ---------------------------------------------------------
            client.moveByVelocityBodyFrameAsync(
                vx=CRUISE_SPEED,
                vy=0,
                vz=float(vz),
                duration=0.1,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0),
                vehicle_name=VEHICLE_NAME
            ).join()

except KeyboardInterrupt:
    print("\n降落...")
    client.reset()