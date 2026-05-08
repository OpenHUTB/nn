"""
该脚本提供了一个与 EasyCarla-RL 环境交互的最小示例。
它遵循标准的 Gym 接口（reset、step），并演示了环境的基本使用方法。
"""

import os
import csv
import pickle
import math
import json
import argparse
from datetime import datetime

import gym
import easycarla
import carla
import random
import numpy as np

try:
    import pygame
except ImportError:
    pygame = None


# 配置环境参数
params = {
    'number_of_vehicles': 20,
    'number_of_walkers': 0,
    'dt': 0.1,  # 两帧之间的时间间隔
    'ego_vehicle_filter': 'vehicle.tesla.model3',  # 用于定义自车的车辆过滤器
    'surrounding_vehicle_spawned_randomly': True,  # 周围车辆是否随机生成
    'port': 2000,  # 连接端口
    'town': 'Town03',  # 要模拟的城市场景
    'max_time_episode': 300,  # 每个 episode 的最大时间步数
    'max_waypoints': 12,  # 最大路点数量
    'visualize_waypoints': True,  # 是否可视化路点
    'desired_speed': 8,  # 期望速度，单位为米/秒
    'max_ego_spawn_times': 200,  # 自车生成的最大尝试次数
    'view_mode': 'top',  # 'top' 表示鸟瞰视角，'follow' 表示第三人称跟随视角
    'traffic': 'off',  # 'on' 表示正常交通灯，'off' 表示始终绿灯并冻结
    'lidar_max_range': 50.0,  # 激光雷达最大感知范围
    'max_nearby_vehicles': 5,  # 可观测的附近车辆最大数量
}


CONTROL_MODE = "autopilot"   # 可选: "autopilot" / "random" / "safe_random" / "manual"
SAVE_EPISODES = True
SAVE_SUMMARY_CSV = True
SAVE_TRAJECTORY_CSV = True
DEBUG_DRAW_EVERY = 5
NUM_EPISODES = 5

# 手动驾驶相关参数
MANUAL_STEER_CACHE = 0.0
MANUAL_QUIT = False

MANUAL_THROTTLE_VALUE = 0.6
MANUAL_BRAKE_VALUE = 0.8
MANUAL_STEER_STEP = 0.04
MANUAL_STEER_DECAY = 0.85


def parse_args():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="EasyCarla-RL demo script")

    parser.add_argument(
        "--mode",
        type=str,
        default=CONTROL_MODE,
        choices=["autopilot", "random", "safe_random", "manual"],
        help="控制模式"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=NUM_EPISODES,
        help="运行 episode 数量"
    )

    parser.add_argument(
        "--town",
        type=str,
        default=params["town"],
        help="CARLA 地图名称"
    )

    parser.add_argument(
        "--vehicles",
        type=int,
        default=params["number_of_vehicles"],
        help="周围车辆数量"
    )

    parser.add_argument(
        "--walkers",
        type=int,
        default=params["number_of_walkers"],
        help="行人数量"
    )

    parser.add_argument(
        "--traffic",
        type=str,
        default=params["traffic"],
        choices=["on", "off"],
        help="交通灯设置"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=params["port"],
        help="CARLA 连接端口"
    )

    parser.add_argument(
        "--save-root",
        type=str,
        default="collected_episodes",
        help="数据保存根目录"
    )

    parser.add_argument(
        "--debug-draw-every",
        type=int,
        default=DEBUG_DRAW_EVERY,
        help="CARLA 画面调试信息绘制间隔"
    )

    parser.add_argument(
        "--no-save-episodes",
        action="store_true",
        help="不保存 episode pkl 数据"
    )

    parser.add_argument(
        "--no-save-summary",
        action="store_true",
        help="不保存 summary.csv"
    )

    parser.add_argument(
        "--no-save-trajectory",
        action="store_true",
        help="不保存 trajectory csv 数据"
    )

    return parser.parse_args()


# 读取命令行参数
args = parse_args()

CONTROL_MODE = args.mode
NUM_EPISODES = args.episodes
DEBUG_DRAW_EVERY = args.debug_draw_every
SAVE_EPISODES = not args.no_save_episodes
SAVE_SUMMARY_CSV = not args.no_save_summary
SAVE_TRAJECTORY_CSV = not args.no_save_trajectory

params["town"] = args.town
params["number_of_vehicles"] = args.vehicles
params["number_of_walkers"] = args.walkers
params["traffic"] = args.traffic
params["port"] = args.port

# 运行批次编号
RUN_ID = datetime.now().strftime("run_%Y%m%d_%H%M%S")


# 数据保存目录
save_root_dir = args.save_root
save_dir = os.path.join(save_root_dir, CONTROL_MODE, RUN_ID)
os.makedirs(save_dir, exist_ok=True)

summary_csv_path = os.path.join(save_dir, "summary.csv")
config_json_path = os.path.join(save_dir, "config.json")


if SAVE_SUMMARY_CSV and not os.path.exists(summary_csv_path):
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode_id",
            "control_mode",
            "steps",
            "total_reward",
            "avg_reward",
            "total_cost",
            "avg_cost",
            "end_reason",
            "collision",
            "off_road",
            "avg_speed",
            "max_speed",
            "min_speed",
            "total_distance"
        ])


def save_config_json(save_path):
    """
    保存本次运行的实验配置。
    """
    config = {
        "run_id": RUN_ID,
        "control_mode": CONTROL_MODE,
        "num_episodes": NUM_EPISODES,
        "save_episodes": SAVE_EPISODES,
        "save_summary_csv": SAVE_SUMMARY_CSV,
        "save_trajectory_csv": SAVE_TRAJECTORY_CSV,
        "debug_draw_every": DEBUG_DRAW_EVERY,
        "save_root_dir": save_root_dir,
        "save_dir": save_dir,
        "manual_control": {
            "manual_throttle_value": MANUAL_THROTTLE_VALUE,
            "manual_brake_value": MANUAL_BRAKE_VALUE,
            "manual_steer_step": MANUAL_STEER_STEP,
            "manual_steer_decay": MANUAL_STEER_DECAY
        },
        "command_args": {
            "mode": args.mode,
            "episodes": args.episodes,
            "town": args.town,
            "vehicles": args.vehicles,
            "walkers": args.walkers,
            "traffic": args.traffic,
            "port": args.port,
            "save_root": args.save_root,
            "debug_draw_every": args.debug_draw_every,
            "no_save_episodes": args.no_save_episodes,
            "no_save_summary": args.no_save_summary,
            "no_save_trajectory": args.no_save_trajectory
        },
        "params": params
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def handle_reset_result(reset_result):
    """
    兼容不同版本 Gym 的 reset 返回值。
    """
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result
        info = {}

    return obs, info


def handle_step_result(step_result):
    """
    兼容不同版本 step 的返回值。
    """
    if len(step_result) == 5:
        next_obs, reward, cost, done, info = step_result

    elif len(step_result) == 6:
        next_obs, reward, cost, terminated, truncated, info = step_result
        done = terminated or truncated

    else:
        raise ValueError(f"Unexpected step return length: {len(step_result)}")

    return next_obs, reward, cost, done, info


def flatten_obs(obs):
    """
    将字典形式的观测数据转换为一维向量。
    """
    state = np.concatenate([
        np.asarray(obs["ego_state"], dtype=np.float32).reshape(-1),
        np.asarray(obs["lane_info"], dtype=np.float32).reshape(-1),
        np.asarray(obs["lidar"], dtype=np.float32).reshape(-1),
        np.asarray(obs["nearby_vehicles"], dtype=np.float32).reshape(-1),
        np.asarray(obs["waypoints"], dtype=np.float32).reshape(-1),
    ])

    return state


def safe_info_dict(info):
    """
    保存常用的 info 字段。
    """
    return {
        "is_collision": bool(info.get("is_collision", False)),
        "is_off_road": bool(info.get("is_off_road", False)),
    }


def get_ego_pose(env):
    """
    获取自车的位置和朝向信息。
    """
    ego_transform = env.ego.get_transform()
    ego_location = ego_transform.location
    ego_rotation = ego_transform.rotation

    ego_pose = {
        "location": {
            "x": float(ego_location.x),
            "y": float(ego_location.y),
            "z": float(ego_location.z),
        },
        "rotation": {
            "pitch": float(ego_rotation.pitch),
            "yaw": float(ego_rotation.yaw),
            "roll": float(ego_rotation.roll),
        }
    }

    return ego_pose


def calculate_distance(location_1, location_2):
    """
    计算两个位置之间的平面距离。
    """
    if location_1 is None or location_2 is None:
        return 0.0

    dx = location_2["x"] - location_1["x"]
    dy = location_2["y"] - location_1["y"]

    return float(math.sqrt(dx * dx + dy * dy))


def save_trajectory_csv(save_path, trajectory_data):
    """
    保存车辆轨迹数据。
    """
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "step",
            "x",
            "y",
            "z",
            "pitch",
            "yaw",
            "roll",
            "speed",
            "reward",
            "cost",
            "step_distance",
            "total_distance",
            "collision",
            "off_road"
        ])

        for row in trajectory_data:
            writer.writerow([
                row["step"],
                row["x"],
                row["y"],
                row["z"],
                row["pitch"],
                row["yaw"],
                row["roll"],
                row["speed"],
                row["reward"],
                row["cost"],
                row["step_distance"],
                row["total_distance"],
                row["collision"],
                row["off_road"]
            ])


def init_manual_control():
    """
    初始化手动驾驶控制窗口。
    """
    if CONTROL_MODE != "manual":
        return

    if pygame is None:
        raise ImportError(
            "当前环境没有安装 pygame，无法使用 manual 手动驾驶模式。"
            "可以先执行：pip install pygame"
        )

    pygame.init()
    pygame.display.set_mode((500, 160))
    pygame.display.set_caption(
        "CARLA Manual Control - W/S/A/D or Arrow Keys, Space Brake, ESC Quit"
    )

    print("Manual control started.")
    print("W / Up       : throttle")
    print("S / Down     : brake")
    print("A / Left     : steer left")
    print("D / Right    : steer right")
    print("Space        : full brake")
    print("ESC          : quit")


def close_manual_control():
    """
    关闭手动驾驶控制窗口。
    """
    if pygame is not None and pygame.get_init():
        pygame.quit()


def get_manual_action():
    """
    通过键盘获取手动驾驶动作。
    动作格式为 [throttle, steer, brake]。
    """
    global MANUAL_STEER_CACHE
    global MANUAL_QUIT

    if pygame is None:
        raise RuntimeError("pygame is not available, cannot use manual mode.")

    pygame.event.pump()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            MANUAL_QUIT = True

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                MANUAL_QUIT = True

    keys = pygame.key.get_pressed()

    throttle = 0.0
    brake = 0.0

    if keys[pygame.K_w] or keys[pygame.K_UP]:
        throttle = MANUAL_THROTTLE_VALUE

    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        brake = MANUAL_BRAKE_VALUE

    if keys[pygame.K_SPACE]:
        throttle = 0.0
        brake = 1.0

    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        MANUAL_STEER_CACHE -= MANUAL_STEER_STEP

    elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        MANUAL_STEER_CACHE += MANUAL_STEER_STEP

    else:
        MANUAL_STEER_CACHE *= MANUAL_STEER_DECAY

    MANUAL_STEER_CACHE = float(np.clip(MANUAL_STEER_CACHE, -1.0, 1.0))

    action = [throttle, MANUAL_STEER_CACHE, brake]

    pygame.display.set_caption(
        f"CARLA Manual Control | throttle={throttle:.2f}, "
        f"steer={MANUAL_STEER_CACHE:.2f}, brake={brake:.2f}"
    )

    return action


def set_control_mode(env, control_mode):
    """
    设置车辆控制模式。
    """
    if control_mode == "autopilot":
        env.ego.set_autopilot(True)

    elif control_mode in ["random", "safe_random", "manual"]:
        env.ego.set_autopilot(False)

    else:
        raise ValueError(f"Unsupported CONTROL_MODE: {control_mode}")


# 定义一个简单的动作策略
def get_action(env, obs, control_mode="autopilot"):
    if control_mode == "autopilot":
        control = env.ego.get_control()
        action = [control.throttle, control.steer, control.brake]

    elif control_mode == "random":
        throttle = random.uniform(0.0, 1.0)
        steer = random.uniform(-0.6, 0.6)
        brake = random.uniform(0.0, 0.3)
        action = [throttle, steer, brake]

    elif control_mode == "safe_random":
        speed = obs["ego_state"][3]

        throttle = random.uniform(0.2, 0.8)
        steer = random.uniform(-0.4, 0.4)
        brake = 0.0

        if speed > params["desired_speed"] * 1.5:
            throttle = 0.0
            brake = 0.4

        elif speed < params["desired_speed"] * 0.3:
            throttle = random.uniform(0.5, 0.9)
            brake = 0.0

        action = [throttle, steer, brake]

    elif control_mode == "manual":
        action = get_manual_action()

    else:
        raise ValueError(f"Unsupported CONTROL_MODE: {control_mode}")

    action = np.array(action, dtype=np.float32)
    action[0] = np.clip(action[0], 0.0, 1.0)
    action[1] = np.clip(action[1], -1.0, 1.0)
    action[2] = np.clip(action[2], 0.0, 1.0)

    return action


# 保存本次运行配置
save_config_json(config_json_path)
print(f"Run ID: {RUN_ID}")
print(f"Save directory: {save_dir}")
print(f"Config saved to: {config_json_path}")


# 创建环境
env = gym.make('carla-v0', params=params)


# 初始化手动驾驶窗口
init_manual_control()


# 与环境交互
try:
    for episode in range(NUM_EPISODES):
        reset_result = env.reset()
        obs, info = handle_reset_result(reset_result)

        set_control_mode(env, CONTROL_MODE)

        done = False
        total_reward = 0.0
        total_cost = 0.0
        episode_data = []
        trajectory_data = []
        end_reason = "unknown"

        speed_list = []
        episode_collision = False
        episode_off_road = False

        total_distance = 0.0
        previous_pose = get_ego_pose(env)
        previous_location = previous_pose["location"]

        while not done:
            action = get_action(env, obs, CONTROL_MODE)

            if CONTROL_MODE == "manual" and MANUAL_QUIT:
                print("[Manual] Exit signal received.")
                end_reason = "manual_quit"
                break

            try:
                step_result = env.step(action)
                next_obs, reward, cost, done, info = handle_step_result(step_result)

            except Exception as e:
                print(f"[Error] Carla step failed: {e}")
                end_reason = "step_error"
                break

            reward = float(reward)
            cost = float(cost)

            speed = float(next_obs['ego_state'][3])
            collision = bool(info.get('is_collision', False))
            off_road = bool(info.get('is_off_road', False))
            ego_pose = get_ego_pose(env)

            current_location = ego_pose["location"]
            step_distance = calculate_distance(previous_location, current_location)
            total_distance += step_distance
            previous_location = current_location

            speed_list.append(speed)
            episode_collision = episode_collision or collision
            episode_off_road = episode_off_road or off_road

            if done:
                if collision:
                    end_reason = "collision"
                elif off_road:
                    end_reason = "off_road"
                elif env.time_step >= params["max_time_episode"]:
                    end_reason = "timeout"
                else:
                    end_reason = "done_other"

            transition = {
                "obs": obs,
                "state": flatten_obs(obs),
                "action": np.array(action, dtype=np.float32),
                "reward": reward,
                "cost": cost,
                "next_obs": next_obs,
                "next_state": flatten_obs(next_obs),
                "done": bool(done),
                "info": safe_info_dict(info),
                "ego_pose": ego_pose,
                "step_distance": float(step_distance),
                "total_distance": float(total_distance),
            }

            episode_data.append(transition)

            trajectory_data.append({
                "step": int(env.time_step),
                "x": ego_pose["location"]["x"],
                "y": ego_pose["location"]["y"],
                "z": ego_pose["location"]["z"],
                "pitch": ego_pose["rotation"]["pitch"],
                "yaw": ego_pose["rotation"]["yaw"],
                "roll": ego_pose["rotation"]["roll"],
                "speed": speed,
                "reward": reward,
                "cost": cost,
                "step_distance": float(step_distance),
                "total_distance": float(total_distance),
                "collision": collision,
                "off_road": off_road,
            })

            if env.time_step % 10 == 0 or done:
                print(
                    f"Step: {env.time_step:4d} | "
                    f"Speed: {speed:6.2f} m/s | "
                    f"Reward: {reward:7.2f} | "
                    f"Cost: {cost:6.2f} | "
                    f"Distance: {total_distance:7.2f} m | "
                    f"Done: {done}"
                )

            if env.time_step % DEBUG_DRAW_EVERY == 0 or done:
                ego_location = env.ego.get_transform().location
                text_location = carla.Location(
                    x=ego_location.x,
                    y=ego_location.y,
                    z=ego_location.z + 2.5
                )

                env.world.debug.draw_string(
                    text_location,
                    f"Mode: {CONTROL_MODE} | "
                    f"Speed: {speed:.2f} m/s | "
                    f"Distance: {total_distance:.2f} m | "
                    f"Reward: {reward:.2f} | "
                    f"Cost: {cost:.2f} | "
                    f"Collision: {collision} | "
                    f"OffRoad: {off_road}",
                    draw_shadow=False,
                    color=carla.Color(0, 255, 0),
                    life_time=0.12,
                    persistent_lines=False
                )

            obs = next_obs
            total_reward += reward
            total_cost += cost

        if len(speed_list) > 0:
            avg_speed = float(np.mean(speed_list))
            max_speed = float(np.max(speed_list))
            min_speed = float(np.min(speed_list))
        else:
            avg_speed = 0.0
            max_speed = 0.0
            min_speed = 0.0

        if len(episode_data) > 0:
            avg_reward = float(total_reward / len(episode_data))
            avg_cost = float(total_cost / len(episode_data))
        else:
            avg_reward = 0.0
            avg_cost = 0.0

        if SAVE_EPISODES:
            episode_record = {
                "episode_id": episode,
                "control_mode": CONTROL_MODE,
                "run_id": RUN_ID,
                "params": params,
                "total_reward": float(total_reward),
                "avg_reward": avg_reward,
                "total_cost": float(total_cost),
                "avg_cost": avg_cost,
                "num_steps": len(episode_data),
                "end_reason": end_reason,
                "collision": bool(episode_collision),
                "off_road": bool(episode_off_road),
                "avg_speed": avg_speed,
                "max_speed": max_speed,
                "min_speed": min_speed,
                "total_distance": float(total_distance),
                "data": episode_data,
            }

            save_path = os.path.join(save_dir, f"episode_{episode:03d}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(episode_record, f)

            print(f"Episode data saved to: {save_path}")

        if SAVE_TRAJECTORY_CSV:
            trajectory_csv_path = os.path.join(save_dir, f"trajectory_episode_{episode:03d}.csv")
            save_trajectory_csv(trajectory_csv_path, trajectory_data)
            print(f"Trajectory data saved to: {trajectory_csv_path}")

        if SAVE_SUMMARY_CSV:
            with open(summary_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode,
                    CONTROL_MODE,
                    len(episode_data),
                    float(total_reward),
                    avg_reward,
                    float(total_cost),
                    avg_cost,
                    end_reason,
                    bool(episode_collision),
                    bool(episode_off_road),
                    avg_speed,
                    max_speed,
                    min_speed,
                    float(total_distance)
                ])

        print(
            f"Episode {episode} finished. "
            f"Steps: {len(episode_data)} | "
            f"Total reward: {total_reward:.2f} | "
            f"Avg reward: {avg_reward:.2f} | "
            f"Total cost: {total_cost:.2f} | "
            f"Avg cost: {avg_cost:.2f} | "
            f"End reason: {end_reason} | "
            f"Avg speed: {avg_speed:.2f} m/s | "
            f"Max speed: {max_speed:.2f} m/s | "
            f"Distance: {total_distance:.2f} m"
        )

        if CONTROL_MODE == "manual" and MANUAL_QUIT:
            break

finally:
    close_manual_control()
    env.close()