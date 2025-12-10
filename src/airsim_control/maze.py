import airsim
import numpy as np
import time
import math

# --- 配置 ---
VEHICLE_NAME = "Drone_1"
LIDAR_NAME = "lidar_1"

# 飞行参数
TARGET_HEIGHT = -1.5
CRUISE_SPEED = 2.5  # 稍微提速
TURN_SPEED = 90.0  # 转快点，别磨叽
STOP_DIST = 3.5  # 刹车距离
PASS_DIST = 4.5  # 判定通行的距离
GRID_SIZE = 2.0  # 记忆格大小

# 极远距离 (视为出口)
EXIT_DIST_THRESHOLD = 15.0

# 可视化开关
VISUALIZE = True


# --- 记忆模块 ---
class MemoryMap:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.visited = set()
        self.forbidden = set()

    def _to_grid(self, x, y):
        return (round(x / self.grid_size), round(y / self.grid_size))

    def mark_visited(self, pos_x, pos_y, client):
        gx, gy = self._to_grid(pos_x, pos_y)
        if (gx, gy) in self.forbidden: return
        if (gx, gy) not in self.visited:
            self.visited.add((gx, gy))
            if VISUALIZE:
                client.simPlotPoints([airsim.Vector3r(gx * self.grid_size, gy * self.grid_size, -1.5)],
                                     color_rgba=[0.0, 0.0, 1.0, 1.0], size=15, is_persistent=True)

    def mark_forbidden(self, pos_x, pos_y, client):
        gx, gy = self._to_grid(pos_x, pos_y)
        if (gx, gy) not in self.forbidden:
            self.forbidden.add((gx, gy))
            if VISUALIZE:
                client.simPlotPoints([airsim.Vector3r(gx * self.grid_size, gy * self.grid_size, -1.5)],
                                     color_rgba=[0.0, 0.0, 0.0, 1.0], size=30, is_persistent=True)

    def check_status(self, pos_x, pos_y):
        gx, gy = self._to_grid(pos_x, pos_y)
        if (gx, gy) in self.forbidden: return 2

        # 范围检查：如果目标点或其相邻点去过，都算去过（模糊匹配）
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (gx + dx, gy + dy) in self.visited:
                    return 1
        return 0


# 初始化
memory = MemoryMap(GRID_SIZE)
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, vehicle_name=VEHICLE_NAME)
client.armDisarm(True, vehicle_name=VEHICLE_NAME)

print("🚀 起飞中...")
client.takeoffAsync(vehicle_name=VEHICLE_NAME).join()
client.moveToZAsync(TARGET_HEIGHT, 1, vehicle_name=VEHICLE_NAME).join()

print("\n=== 终极寻路: 直觉 + 出口诱导 ===")


def get_lidar_info():
    """获取前、左、右距离"""
    lidar_data = client.getLidarData(lidar_name=LIDAR_NAME, vehicle_name=VEHICLE_NAME)
    if not lidar_data or len(lidar_data.point_cloud) < 3: return 99, 99, 99

    points = np.array(lidar_data.point_cloud, dtype=np.float32)
    points = np.reshape(points, (int(points.shape[0] / 3), 3))
    valid = points[(points[:, 2] > -0.5) & (points[:, 2] < 0.5)]
    if len(valid) == 0: return 99, 99, 99

    f_mask = (valid[:, 0] > 0) & (np.abs(valid[:, 1]) < 1.0)
    l_mask = (valid[:, 1] < -1.0) & (np.abs(valid[:, 0]) < 1.0)
    r_mask = (valid[:, 1] > 1.0) & (np.abs(valid[:, 0]) < 1.0)

    f_d = np.min(valid[f_mask][:, 0]) if np.any(f_mask) else 99
    l_d = np.min(np.linalg.norm(valid[l_mask][:, :2], axis=1)) if np.any(l_mask) else 99
    r_d = np.min(np.linalg.norm(valid[r_mask][:, :2], axis=1)) if np.any(r_mask) else 99

    return f_d, l_d, r_d


def get_global_yaw():
    o = client.simGetVehiclePose(vehicle_name=VEHICLE_NAME).orientation
    return math.degrees(
        math.atan2(2.0 * (o.w_val * o.z_val + o.x_val * o.y_val), 1.0 - 2.0 * (o.y_val * o.y_val + o.z_val * o.z_val)))


def turn_rel(angle):
    print(f"   ↪️ 转向 {angle}°...")
    client.moveByVelocityAsync(0, 0, 0, abs(angle) / TURN_SPEED,
                               drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                               yaw_mode=airsim.YawMode(is_rate=True,
                                                       yaw_or_rate=float(TURN_SPEED if angle > 0 else -TURN_SPEED)),
                               vehicle_name=VEHICLE_NAME).join()
    client.moveByVelocityAsync(0, 0, 0, 0.2, vehicle_name=VEHICLE_NAME).join()


def check_direction_score(pos, curr_yaw, angle, lidar_dist):
    """评分系统"""
    # 1. 物理阻挡
    if lidar_dist < PASS_DIST:
        return -1000, "🧱 阻挡"

    # 2. 【核心优化】出口检测
    # 如果雷达距离极远(>15米)，说明前面是开阔地(出口)，给予超高分！
    if lidar_dist > EXIT_DIST_THRESHOLD:
        client.simPlotPoints([airsim.Vector3r(pos.x_val, pos.y_val, -1.5)], color_rgba=[1.0, 1.0, 0.0, 1.0], size=30,
                             duration=5.0)
        return 10000, "🎉 出口/开阔地"

    # 3. 记忆检查
    rad = math.radians(curr_yaw + angle)
    check_dist = 4.0
    target_x = pos.x_val + math.cos(rad) * check_dist
    target_y = pos.y_val + math.sin(rad) * check_dist

    status_code = memory.check_status(target_x, target_y)

    if status_code == 2:  # 死路
        return -1000, "⚫ 死路"

    elif status_code == 1:  # 老路
        # 调试：红点
        client.simPlotPoints([airsim.Vector3r(target_x, target_y, -1.5)], color_rgba=[1.0, 0.0, 0.0, 1.0], size=10,
                             duration=2.0)
        return -50, "👣 老路"

    else:  # 新路
        # 调试：绿点
        client.simPlotPoints([airsim.Vector3r(target_x, target_y, -1.5)], color_rgba=[0.0, 1.0, 0.0, 1.0], size=20,
                             duration=2.0)
        return 100, "✨ 新路"


def scan_and_decide():
    print("\n🛑 决策中...")
    client.moveByVelocityAsync(0, 0, 0, 0.5, vehicle_name=VEHICLE_NAME).join()

    pos = client.simGetVehiclePose(vehicle_name=VEHICLE_NAME).position
    curr_yaw = get_global_yaw()
    f_d, l_d, r_d = get_lidar_info()

    options = [
        {"angle": 0, "dist": f_d, "name": "前方"},
        {"angle": -90, "dist": l_d, "name": "左侧"},
        {"angle": 90, "dist": r_d, "name": "右侧"}
    ]

    candidates = []

    print("   📊 评分:")
    for opt in options:
        score, status = check_direction_score(pos, curr_yaw, opt["angle"], opt["dist"])

        # 只有非墙壁才加入
        if score > -500:
            candidates.append({
                "angle": opt["angle"],
                "score": score,
                "name": opt["name"],
                "dist": opt["dist"]
            })
            print(f"      -> {opt['name']}: {status} ({score})")

    if len(candidates) > 0:
        # 排序逻辑优化：
        # 1. 分数高的优先
        # 2. 【核心优化】分数相同时，优先选角度为0的(直行)！避免左右乱转
        # 3. 最后选距离远的
        # 我们用 tuple 排序: (score, is_straight, dist)
        # angle == 0 转换为 1 (是直行), 否则 0

        candidates.sort(key=lambda x: (x["score"], 1 if x["angle"] == 0 else 0, x["dist"]), reverse=True)

        best = candidates[0]
        print(f"✅ 决定: {best['name']}")

        if best["angle"] != 0:
            turn_rel(best["angle"])

        return True  # 找到了路

    else:
        print("⚠️ 全是死路! 掉头并封锁")
        memory.mark_forbidden(pos.x_val, pos.y_val, client)
        turn_rel(180)
        return False  # 被迫掉头


try:
    # 强制冷却时间 (防止刚转完头又觉得不对劲)
    cooldown_until = 0

    while True:
        # 记录足迹
        pos = client.simGetVehiclePose(vehicle_name=VEHICLE_NAME).position
        memory.mark_visited(pos.x_val, pos.y_val, client)

        f_d, l_d, r_d = get_lidar_info()

        # 状态判定
        is_stuck = f_d < STOP_DIST
        # 只有当侧面非常宽敞(>5m)，且没在冷却期内，才视为岔路
        is_junction = (l_d > 5.0 or r_d > 5.0) and time.time() > cooldown_until

        # --- 优先级 1: 看到出口 (Exit) ---
        # 如果前方一片空旷 (>15米)，说明要出去了，无视所有逻辑直接冲
        if f_d > EXIT_DIST_THRESHOLD:
            print(f"\r[🎉 发现出口!] 前方开阔 {f_d:.1f}m - 全速前进!", end="")
            client.moveByVelocityBodyFrameAsync(CRUISE_SPEED * 1.5, 0, 0, 0.1,
                                                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0),
                                                vehicle_name=VEHICLE_NAME).join()
            continue  # 跳过后面所有逻辑

        # --- 优先级 2: 遇阻 ---
        if is_stuck:
            print(f"\r[🛑 遇阻] 前方 {f_d:.1f}m", end="")
            scan_and_decide()
            # 决策完后，给 3秒 冷却时间，让它先飞离路口，别原地纠结
            cooldown_until = time.time() + 3.0

        # --- 优先级 3: 岔路 ---
        elif is_junction:
            print(f"\r[✨ 岔路] 左:{l_d:.1f}m 右:{r_d:.1f}m", end="")
            print(" -> 决策...")
            # 往前送 2米
            client.moveByVelocityBodyFrameAsync(CRUISE_SPEED, 0, 0, 1.5,
                                                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0),
                                                vehicle_name=VEHICLE_NAME).join()

            scan_and_decide()
            cooldown_until = time.time() + 3.0

        # --- 优先级 4: 巡航 ---
        else:
            print(f"\r[🚀 巡航] 前:{f_d:.1f}m   ", end="", flush=True)

            z_curr = client.simGetVehiclePose(vehicle_name=VEHICLE_NAME).position.z_val
            vz = (TARGET_HEIGHT - z_curr) * 1.0

            # 简单的居中
            vy = 0
            if l_d < 2.0 and r_d < 2.0:
                vy = (l_d - r_d) * 0.5
                vy = np.clip(vy, -1.0, 1.0)

            client.moveByVelocityBodyFrameAsync(
                vx=CRUISE_SPEED, vy=float(vy), vz=float(vz), duration=0.1,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0),
                vehicle_name=VEHICLE_NAME
            ).join()

except KeyboardInterrupt:
    print("\n降落...")
    client.reset()