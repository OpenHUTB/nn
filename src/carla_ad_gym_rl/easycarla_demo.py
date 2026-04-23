"""
该脚本提供了一个与 EasyCarla-RL 环境交互的最小示例。
它遵循标准的 Gym 接口（reset、step），并演示了环境的基本使用方法。
"""

import os
import csv
import pickle

import gym
import easycarla
import carla
import random
import numpy as np

# 配置环境参数
params = {
    'number_of_vehicles': 20,
    'number_of_walkers': 0,
    'dt': 0.1,  # 两帧之间的时间间隔
    'ego_vehicle_filter': 'vehicle.tesla.model3',  # 用于定义自车的车辆过滤器
    'surrounding_vehicle_spawned_randomly': True, # 周围车辆是否随机生成（True）或手动设置（False）
    'port': 2000,  # 连接端口
    'town': 'Town03',  # 要模拟的城市场景
    'max_time_episode': 300,  # 每个 episode 的最大时间步数
    'max_waypoints': 12,  # 最大路点数量
    'visualize_waypoints': True,  # 是否可视化路点（默认：True）
    'desired_speed': 8,  # 期望速度（米/秒）
    'max_ego_spawn_times': 200,  # 自车生成的最大尝试次数
    'view_mode' : 'top',  # 'top' 表示鸟瞰视角，'follow' 表示第三人称跟随视角
    'traffic': 'off',  # 'on' 表示正常交通灯，'off' 表示始终绿灯并冻结
    'lidar_max_range': 50.0,  # 激光雷达最大感知范围（米）
    'max_nearby_vehicles': 5,  # 可观测的附近车辆最大数量
}

CONTROL_MODE = "autopilot"   # 可选: "autopilot" / "random"
SAVE_EPISODES = True
SAVE_SUMMARY_CSV = True
DEBUG_DRAW_EVERY = 5

# 创建环境
env = gym.make('carla-v0', params=params)

# 数据保存目录
save_dir = "collected_episodes"
os.makedirs(save_dir, exist_ok=True)

summary_csv_path = os.path.join(save_dir, "summary.csv")

if SAVE_SUMMARY_CSV and not os.path.exists(summary_csv_path):
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode_id",
            "control_mode",
            "steps",
            "total_reward",
            "total_cost",
            "end_reason"
        ])

reset_result = env.reset()
if isinstance(reset_result, tuple):
    obs, info = reset_result
else:
    obs = reset_result
    info = {}

# 定义一个简单的动作策略
def get_action(env, obs, control_mode="autopilot"):
    if control_mode == "autopilot":
        env.ego.set_autopilot(True)
        control = env.ego.get_control()
        return [control.throttle, control.steer, control.brake]

    elif control_mode == "random":
        env.ego.set_autopilot(False)
        throttle = random.uniform(0.0, 1.0)
        steer = random.uniform(-0.6, 0.6)
        brake = random.uniform(0.0, 0.3)
        return [throttle, steer, brake]

    else:
        raise ValueError(f"Unsupported CONTROL_MODE: {control_mode}")

# 与环境交互
try:
    for episode in range(5):  # 运行 5 个 episode
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}

        done = False
        total_reward = 0
        total_cost = 0.0
        episode_data = []
        end_reason = "unknown"

        while not done:
            action = get_action(env, obs, CONTROL_MODE)

            try:
                step_result = env.step(action)
            except Exception as e:
                print(f"[Error] Carla step failed: {e}")
                end_reason = "step_error"
                break

            if len(step_result) == 5:
                next_obs, reward, cost, done, info = step_result
            elif len(step_result) == 6:
                next_obs, reward, cost, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                raise ValueError(f"Unexpected step return length: {len(step_result)}")
            
            if done:
                if info.get("is_collision", False):
                    end_reason = "collision"
                elif info.get("is_off_road", False):
                    end_reason = "off_road"
                elif env.time_step >= params["max_time_episode"]:
                    end_reason = "timeout"
                else:
                    end_reason = "done_other"
            
            transition = {
                "obs": obs,
                "action": np.array(action, dtype=np.float32),
                "reward": float(reward),
                "cost": float(cost),
                "next_obs": next_obs,
                "done": bool(done),
                "info": info,
            }
            episode_data.append(transition)

            # 每隔固定步数输出一次当前 step 的奖励、代价和结束状态
            if env.time_step % 10 == 0 or done:
                print(
                    f"Step: {env.time_step:4d} | "
                    f"Reward: {reward:7.2f} | "
                    f"Cost: {cost:6.2f} | "
                    f"Done: {done}"
                )

            # 提取车辆当前速度及运行状态，并在 CARLA 画面中显示监控信息
            speed = next_obs['ego_state'][3]
            collision = info.get('is_collision', False)
            off_road = info.get('is_off_road', False)

            ego_location = env.ego.get_transform().location
            text_location = carla.Location(
                x=ego_location.x,
                y=ego_location.y,
                z=ego_location.z + 2.5
            )

            if env.time_step % DEBUG_DRAW_EVERY == 0 or done:
                speed = next_obs['ego_state'][3]
                collision = info.get('is_collision', False)
                off_road = info.get('is_off_road', False)

                ego_location = env.ego.get_transform().location
                text_location = carla.Location(
                    x=ego_location.x,
                    y=ego_location.y,
                    z=ego_location.z + 2.5
                )

                env.world.debug.draw_string(
                    text_location,
                    f"Mode: {CONTROL_MODE} | Speed: {speed:.2f} m/s | Reward: {reward:.2f} | Cost: {cost:.2f} | Collision: {collision} | OffRoad: {off_road}",
                    draw_shadow=False,
                    color=carla.Color(0, 255, 0),
                    life_time=0.12,
                    persistent_lines=False
                )

            obs = next_obs
            total_reward += reward
            total_cost += cost

        if SAVE_EPISODES:
            episode_record = {
                "episode_id": episode,
                "control_mode": CONTROL_MODE,
                "total_reward": float(total_reward),
                "total_cost": float(total_cost),
                "num_steps": len(episode_data),
                "end_reason": end_reason,
                "data": episode_data,
            }

            save_path = os.path.join(save_dir, f"episode_{episode:03d}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(episode_record, f)

            print(f"Episode data saved to: {save_path}")

        if SAVE_SUMMARY_CSV:
                    with open(summary_csv_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            episode,
                            CONTROL_MODE,
                            len(episode_data),
                            float(total_reward),
                            float(total_cost),
                            end_reason
                        ])

        print(
            f"Episode {episode} finished. "
            f"Steps: {len(episode_data)} | "
            f"Total reward: {total_reward:.2f} | "
            f"Total cost: {total_cost:.2f} | "
            f"End reason: {end_reason}"
        )

finally:
    env.close()