import numpy as np
from config import CONFIG

low_speed_timer = 0

min_speed = CONFIG["reward_params"]["min_speed"]
max_speed = CONFIG["reward_params"]["max_speed"]
target_speed = CONFIG["reward_params"]["target_speed"]
max_distance = CONFIG["reward_params"]["max_distance"]
max_std_center_lane = CONFIG["reward_params"]["max_std_center_lane"]
max_angle_center_lane = CONFIG["reward_params"]["max_angle_center_lane"]
penalty_reward = CONFIG["reward_params"]["penalty_reward"]
early_stop = CONFIG["reward_params"]["early_stop"]
reward_functions = {}


def normalize_angle(angle):
    """将角度归一化到 [-180, 180]，修复跨0°/360°的bug"""
    return (angle + 180) % 360 - 180


def create_reward_fn(reward_fn):
    def func(env):
        global low_speed_timer
        terminal_reason = "Running..."
        if early_stop:
            low_speed_timer += 1.0 / env.fps
            speed = env.get_vehicle_lon_speed()

            # 修复：速度恢复时重置timer，只有"连续"低速才终止
            if speed >= 3.0:
                low_speed_timer = 0.0

            if low_speed_timer > 3.0 and speed < 3.0 and env.current_waypoint_index >= 0:
                env.terminate = True
                terminal_reason = "Vehicle stopped"

            if env.distance_from_center > max_distance:
                env.terminate = True
                terminal_reason = "Off-track"

            if max_speed > 0 and speed > max_speed:
                env.terminate = True
                terminal_reason = "Too fast"

        # Calculate reward
        reward = 0
        if not env.terminate:
            reward += reward_fn(env)
        else:
            low_speed_timer = 0.0
            # 优化：存活越久终止惩罚越大，鼓励长期存活
            reward += max(-2.0, -0.02 * env.current_step)
            print(f"{env.episode_idx}| Terminal: ", terminal_reason)

        if env.success_state:
            print(f"{env.episode_idx}| Success")

        env.extra_info.extend([
            terminal_reason,
            ""
        ])
        return reward

    return func


def reward_fn5(env):
    """
    加权求和 + 不动惩罚 + 前进奖励：
    - 速度奖励：平滑过渡，不动扣分
    - 居中奖励：二次衰减，鼓励精确居中
    - 角度奖励：修复跨0°bug
    - 前进奖励：经过waypoint额外加分
    """
    veh_angle = env.vehicle.get_transform().rotation.yaw
    wayp_angle = env.current_waypoint.transform.rotation.yaw
    # 修复：用归一化角度，避免359°和1°计算出358°
    angle = abs(normalize_angle(wayp_angle - veh_angle))
    speed_kmh = env.get_vehicle_lon_speed()

    # 速度奖励：平滑过渡，消除跳变
    if speed_kmh < 5.0:
        speed_reward = -0.2 + 0.04 * speed_kmh  # [0,5] → [-0.2, 0.0]
    elif speed_kmh < min_speed:
        speed_reward = (speed_kmh - 5.0) / (min_speed - 5.0) * 0.8 + 0.2  # [5,min] → [0.2, 1.0]
    elif speed_kmh > target_speed:
        speed_reward = max(1.0 - (speed_kmh - target_speed) / (max_speed - target_speed), 0.0)
    else:
        speed_reward = 1.0

    # 居中因子：二次衰减，靠近中心时梯度更敏感
    center_ratio = min(env.distance_from_center / max_distance, 1.0)
    centering_factor = max(1.0 - center_ratio ** 0.5, 0.0)

    angle_factor = max(1.0 - angle / max_angle_center_lane, 0.0)

    # 前进奖励：经过waypoint额外加分
    waypoint_reward = (env.current_waypoint_index - env.prev_waypoint_index) * 0.5

    reward = 0.35 * speed_reward + 0.25 * centering_factor + 0.20 * angle_factor + waypoint_reward
    return reward


reward_functions["reward_fn5"] = create_reward_fn(reward_fn5)


def reward_fn_waypoints(env):
    """
    waypoint奖励 + 平滑速度惩罚：
    - 经过waypoint得1.0分
    - 速度惩罚平滑过渡
    - 居中 + 角度辅助信号
    """
    speed_kmh = env.get_vehicle_lon_speed()

    # 速度惩罚：平滑过渡，不再有跳变
    if speed_kmh < 5.0:
        speed_penalty = -0.3 + 0.04 * speed_kmh  # [0,5] → [-0.3, -0.1]
    elif speed_kmh < min_speed:
        speed_penalty = -0.1 * (min_speed - speed_kmh) / (min_speed - 5.0)  # [5,min] → [-0.1, 0]
    else:
        speed_penalty = 0.0

    waypoint_reward = (env.current_waypoint_index - env.prev_waypoint_index) * 1.0

    # 速度奖励
    if speed_kmh < min_speed:
        speed_reward = speed_kmh / min_speed
    elif speed_kmh > target_speed:
        speed_reward = max(1.0 - (speed_kmh - target_speed) / (max_speed - target_speed), 0.0)
    else:
        speed_reward = 1.0

    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

    # 加入角度信号
    veh_angle = env.vehicle.get_transform().rotation.yaw
    wayp_angle = env.current_waypoint.transform.rotation.yaw
    angle = abs(normalize_angle(wayp_angle - veh_angle))
    angle_factor = max(1.0 - angle / max_angle_center_lane, 0.0)

    reward = waypoint_reward + 0.3 * speed_reward + 0.15 * centering_factor + 0.1 * angle_factor + speed_penalty
    return reward


reward_functions["reward_fn_waypoints"] = create_reward_fn(reward_fn_waypoints)