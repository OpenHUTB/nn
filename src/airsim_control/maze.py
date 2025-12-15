import airsim
import numpy as np
import time
import math

# --- 精细操控版配置 ---
VEHICLE_NAME = "Drone_1"
LIDAR_NAME = "lidar_1"

# 飞行参数 (慢速、精准)
TARGET_HEIGHT = -1.5
CRUISE_SPEED = 1.0  #  速度降至 1.0，防止冲过头
TURN_SPEED = 30.0  # 转向速度
STOP_DIST = 2.0  # 刹车距离
PASS_DIST = 2.0  # 通行门槛 (大于2米就敢进)
GRID_SIZE = 1.5
EXIT_DIST_THRESHOLD = 15.0

SIDE_MARGIN = 1.5  # 左右保持距离

VISUALIZE = True


# --- 记忆模块 (保持不变) ---
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
                                     color_rgba=[0.0, 0.0, 1.0, 1.0], size=10, is_persistent=True)

    def mark_forbidden(self, pos_x, pos_y, client):
        gx, gy = self._to_grid(pos_x, pos_y)
        if (gx, gy) not in self.forbidden:
            self.forbidden.add((gx, gy))
            if VISUALIZE:
                client.simPlotPoints([airsim.Vector3r(gx * self.grid_size, gy * self.grid_size, -1.5)],
                                     color_rgba=[0.0, 0.0, 0.0, 1.0], size=25, is_persistent=True)

    def calculate_path_score(self, start_x, start_y, angle_rad, check_dist):
        steps = int(check_dist / self.grid_size)
        if steps == 0: return 0, False, start_x, start_y
        visited_count = 0
        target_x = start_x + math.cos(angle_rad) * check_dist
        target_y = start_y + math.sin(angle_rad) * check_dist

        for i in range(1, steps + 1):
            d = i * self.grid_size
            tx = start_x + math.cos(angle_rad) * d
            ty = start_y + math.sin(angle_rad) * d
            gx, gy = self._to_grid(tx, ty)
            if (gx, gy) in self.forbidden: return -1000, True, tx, ty
            if (gx, gy) in self.visited: visited_count += 1

        density = visited_count / steps
        score = 100 - (density * 150)
        return score, False, target_x, target_y


# 初始化
memory = MemoryMap(GRID_SIZE)
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, vehicle_name=VEHICLE_NAME)
client.armDisarm(True, vehicle_name=VEHICLE_NAME)

print(" 精细操控模式起飞 (低速抗过冲)...")
client.takeoffAsync(vehicle_name=VEHICLE_NAME).join()
client.moveToZAsync(TARGET_HEIGHT, 1, vehicle_name=VEHICLE_NAME).join()

# 暴力起步 (依然需要，但稍微温柔点)
print("💨 起步...")
client.moveByVelocityBodyFrameAsync(1.5, 0, 0, 2.0,
                                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0),
                                    vehicle_name=VEHICLE_NAME).join()


# --- 控制函数 ---

def calculate_z_correction(current_z):
    z_error = TARGET_HEIGHT - current_z
    vz = z_error * 1.5
    return float(np.clip(vz, -0.8, 0.8))


def calculate_y_correction(l_dist, r_dist):
    if l_dist > SIDE_MARGIN and r_dist > SIDE_MARGIN:
        return 0.0
    vy = 0.0
    if l_dist < SIDE_MARGIN:
        push = (SIDE_MARGIN - l_dist) * 0.8
        vy += push
    if r_dist < SIDE_MARGIN:
        push = (SIDE_MARGIN - r_dist) * 0.8
        vy -= push
    return float(np.clip(vy, -0.8, 0.8))  # 降低横向修正力度


def get_lidar_info():
    lidar_data = client.getLidarData(lidar_name=LIDAR_NAME, vehicle_name=VEHICLE_NAME)
    if not lidar_data or len(lidar_data.point_cloud) < 3: return 99, 99, 99
    points = np.array(lidar_data.point_cloud, dtype=np.float32)
    points = np.reshape(points, (int(points.shape[0] / 3), 3))
    valid = points[(points[:, 2] > -0.5) & (points[:, 2] < 0.5)]
    if len(valid) == 0: return 99, 99, 99

    f_mask = (valid[:, 0] > 0) & (np.abs(valid[:, 1]) < 1.0)
    l_mask = (valid[:, 1] < -1.0) & (np.abs(valid[:, 0]) < 1.0)
    r_mask = (valid[:, 1] > 1.0) & (np.abs(valid[:, 0]) < 1.0)

    valid = points[(points[:, 2] > -0.4) & (points[:, 2] < 0.4)]
    if len(valid) == 0: return 99, 99, 99

    f_mask = (valid[:, 0] > 0) & (np.abs(valid[:, 1]) < 0.6)  # 前方判定变窄
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


def turn_to_angle_gentle(target_angle_rel):
    print(f"   ↪️ 缓慢转向 {target_angle_rel}°")
    start_yaw = get_global_yaw()
    target_yaw = start_yaw + target_angle_rel

    if target_yaw > 180:
        target_yaw -= 360
    elif target_yaw < -180:
        target_yaw += 360

    while True:
        current_yaw = get_global_yaw()
        error = target_yaw - current_yaw
        if error > 180:
            error -= 360
        elif error < -180:
            error += 360
        if abs(error) < 1.0: break  # 精度更高 1.0度

        yaw_rate = np.clip(error * 1.0, -30, 30)  # 转得更慢
        if abs(yaw_rate) < 8: yaw_rate = 8 * np.sign(yaw_rate)

        client.moveByVelocityAsync(0, 0, 0, 0.05,
                                   drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                   yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=float(yaw_rate)),
                                   vehicle_name=VEHICLE_NAME).join()

def scan_and_decide():
    print("\n🛑 决策中...")
    client.moveByVelocityAsync(0, 0, 0, 0.5, vehicle_name=VEHICLE_NAME).join()

    pos = client.simGetVehiclePose(vehicle_name=VEHICLE_NAME).position
    curr_yaw = get_global_yaw()
    f_d, l_d, r_d = get_lidar_info()

def scan_and_decide():
    # ⚡ 急刹！多停一会儿，完全消除惯性
    client.moveByVelocityAsync(0, 0, 0, 1.5, vehicle_name=VEHICLE_NAME).join()

    pos = client.simGetVehiclePose(vehicle_name=VEHICLE_NAME).position
    curr_yaw = get_global_yaw()
    f_d, l_d, r_d = get_lidar_info()

    options = [
        {"angle": 0, "dist": f_d, "name": "前方"},
        {"angle": -90, "dist": l_d, "name": "左侧"},
        {"angle": 90, "dist": r_d, "name": "右侧"}
    ]
    candidates = []
    print("   📊 深度评分:")

    for opt in options:
        if opt["dist"] < PASS_DIST: continue
        if opt["dist"] > EXIT_DIST_THRESHOLD:
            candidates.append({"angle": opt["angle"], "score": 99999, "name": opt["name"], "dist": opt["dist"]})
            continue

        rad = math.radians(curr_yaw + opt["angle"])
        score, is_dead_end, tx, ty = memory.calculate_path_score(pos.x_val, pos.y_val, rad, 10.0)

        # ⚡ 侧路优先逻辑 (DFS Bias)
        if not is_dead_end and score > 0 and opt["angle"] != 0:
            print(f"      -> {opt['name']}: 🚪 岔路优先探索")
            score += 300  # 权重加大，确保一定转

        if not is_dead_end and opt["dist"] > 8.0 and score > 0:
            bonus = min(opt["dist"] * 5, 50)
            score += bonus

        status_text = "⚫ 死路" if is_dead_end else ("✨ 新路" if score > 0 else "👣 老路")
        print(f"      -> {opt['name']}: {status_text} ({score:.1f})")

        if not is_dead_end and VISUALIZE:
            color = [0.0, 1.0, 0.0, 1.0] if score > 0 else [1.0, 0.0, 0.0, 1.0]
            client.simPlotPoints([airsim.Vector3r(tx, ty, -1.5)], color_rgba=color, size=15, duration=2.0)

        if score > -900:
            candidates.append({"angle": opt["angle"], "score": score, "name": opt["name"], "dist": opt["dist"]})

    if candidates:
        candidates.sort(key=lambda x: (x["score"], x["dist"]), reverse=True)
        best = candidates[0]
        print(f"✅ 决定: {best['name']}")

        if best["angle"] != 0:
            turn_to_angle_gentle(best["angle"])

            # ⚡ [关键修复] 转弯后检查：是否对准了墙？
            # 如果转过去了，但前方距离很近 (<1.5m)，说明冲过头了，对准了墙
            # 这时候需要做一个横向平移修正
            time.sleep(0.5)
            check_f, check_l, check_r = get_lidar_info()
            if check_f < 1.5:
                print("   ⚠️ 检测到过冲 (对准墙壁) -> 尝试横向修正")
                # 尝试左右平移看看哪边空
                # 这是一个盲猜逻辑：通常往回退一点(机身反方向平移)能对准路口
                # 假设我们刚刚右转，说明路口在机身右后方，如果对准了墙，说明机身太靠前
                # 我们尝试向后倒一点点 (Body Frame X 负方向)
                client.moveByVelocityBodyFrameAsync(-0.5, 0, 0, 1.0, vehicle_name=VEHICLE_NAME).join()

        print("   💨 缓慢进入...")
        curr_z = client.simGetVehiclePose(vehicle_name=VEHICLE_NAME).position.z_val
        vz_fix = calculate_z_correction(curr_z)

        # 缓慢推进进入
        client.moveByVelocityBodyFrameAsync(CRUISE_SPEED, 0, float(vz_fix), 2.0,
                                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0),
                                            vehicle_name=VEHICLE_NAME).join()
        return True
    else:
        print("⚠️ 绝境! 后撤封锁...")
        client.moveByVelocityBodyFrameAsync(-1.0, 0, 0, 2.0, vehicle_name=VEHICLE_NAME).join()  # 慢速后撤
        rad = math.radians(curr_yaw)
        memory.mark_forbidden(pos.x_val + math.cos(rad) * 2.5, pos.y_val + math.sin(rad) * 2.5, client)
        turn_to_angle_gentle(180)
        return False

        candidates.sort(key=lambda x: (x["score"], 1 if x["angle"] == 0 else 0, x["dist"]), reverse=True)

try:
    cooldown_until = 0
    while True:
        pos = client.simGetVehiclePose(vehicle_name=VEHICLE_NAME).position
        memory.mark_visited(pos.x_val, pos.y_val, client)
        f_d, l_d, r_d = get_lidar_info()

        vz_fix = calculate_z_correction(pos.z_val)
        vy_fix = calculate_y_correction(l_d, r_d)

        is_stuck = f_d < STOP_DIST
        # 只要侧面距离 > 2.5米 (放宽一点)，立刻触发
        is_junction = (l_d > 2.5 or r_d > 2.5) and (time.time() > cooldown_until)

        if f_d > EXIT_DIST_THRESHOLD:
            print(f"\r[🚀 冲刺] 开阔地 {f_d:.1f}m", end="")
            client.moveByVelocityBodyFrameAsync(3.0, 0, float(vz_fix), 0.1,
                                                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0),
                                                vehicle_name=VEHICLE_NAME).join()
            continue

        if is_stuck:
            print(f"\r[🛑 遇阻] {f_d:.1f}m", end="")
            scan_and_decide()
            cooldown_until = time.time() + 2.0

        elif is_junction:
            print(f"\r[🚪 发现路口] 左:{l_d:.1f}m 右:{r_d:.1f}m -> ⚡ 急刹决策", end="")
            # ⚡ 关键修改：取消向前送的动作！立即停车！
            client.moveByVelocityAsync(0, 0, 0, 0.5, vehicle_name=VEHICLE_NAME).join()

            scan_and_decide()
            cooldown_until = time.time() + 3.0

        else:
            print(f"\r[🚀 巡航] H:{pos.z_val:.1f} Y:{vy_fix:.2f}", end="", flush=True)
            client.moveByVelocityBodyFrameAsync(
                vx=CRUISE_SPEED,
                vy=float(vy_fix),
                vz=float(vz_fix),
                duration=0.1,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=0),
                vehicle_name=VEHICLE_NAME
            ).join()

except KeyboardInterrupt:
    print("\n降落...")
    client.reset()