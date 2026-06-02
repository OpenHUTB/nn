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


def create_reward_fn(reward_fn):
    def func(env):
        global low_speed_timer
        terminal_reason = "Running..."
        if early_stop:
            low_speed_timer += 1.0 / env.fps
            speed = env.get_vehicle_lon_speed()
            # 优化：3秒内速度低于3km/h就终止（原来5秒+1km/h太宽松）
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
            reward += -1.0
            print(f"{env.episode_idx}| Terminal: ", terminal_reason)

        if env.success_state:
            print(f"{env.episode_idx}| Success")

        env.extra_info.extend([
            terminal_reason,
            ""
        ])
        return reward

    return func


# 优化后的reward_fn5：加入"不动就罚"机制
def reward_fn5(env):
    """
    加权求和 + 不动惩罚：
    - 速度奖励：不动直接扣分，合理速度得高分
    - 居中奖励：离车道中心越近分越高
    - 角度奖励：与道路方向越对齐分越高
    """
    veh_angle = env.vehicle.get_transform().rotation.yaw
    wayp_angle = env.current_waypoint.transform.rotation.yaw
    angle = abs(wayp_angle - veh_angle)
    speed_kmh = env.get_vehicle_lon_speed()

    # 速度奖励：低于5km/h直接罚分
    if speed_kmh < 5.0:
        speed_reward = -0.2
    elif speed_kmh < min_speed:
        speed_reward = speed_kmh / min_speed * 0.5
    elif speed_kmh > target_speed:
        speed_reward = max(1.0 - (speed_kmh - target_speed) / (max_speed - target_speed), 0.0)
    else:
        speed_reward = 1.0

    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)
    angle_factor = max(1.0 - abs(angle / max_angle_center_lane), 0.0)

    reward = 0.45 * speed_reward + 0.30 * centering_factor + 0.25 * angle_factor
    return reward


reward_functions["reward_fn5"] = create_reward_fn(reward_fn5)


# 优化后的waypoint奖励：更直接的"往前开"信号
def reward_fn_waypoints(env):
    """
    waypoint奖励 + 不动惩罚：
    - 经过waypoint得1.0分（只有往前开才能拿到）
    - 速度太低扣分
    - 居中作为辅助信号
    """
    speed_kmh = env.get_vehicle_lon_speed()

    # 速度惩罚：不动就扣分
    if speed_kmh < 5.0:
        speed_penalty = -0.3
    elif speed_kmh < min_speed:
        speed_penalty = -0.1
    else:
        speed_penalty = 0.0

    # waypoint通过奖励
    waypoint_reward = (env.current_waypoint_index - env.prev_waypoint_index) * 1.0

    # 速度奖励
    if speed_kmh < min_speed:
        speed_reward = speed_kmh / min_speed
    elif speed_kmh > target_speed:
        speed_reward = max(1.0 - (speed_kmh - target_speed) / (max_speed - target_speed), 0.0)
    else:
        speed_reward = 1.0

    # 居中因子
    centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

    reward = waypoint_reward + 0.3 * speed_reward + 0.2 * centering_factor + speed_penalty
    return reward


reward_functions["reward_fn_waypoints"] = create_reward_fn(reward_fn_waypoints)