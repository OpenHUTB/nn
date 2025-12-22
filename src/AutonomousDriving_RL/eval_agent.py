# eval_agent.py
"""
增强版 CARLA 智能体评估器
- 支持单目标 / 多目标导航
- 路径可视化 + 平滑转向 + 动态调速
- 无需重新训练模型
"""

import argparse
import numpy as np
import carla
from stable_baselines3 import PPO
from carla_env.carla_env_multi_obs import CarlaEnvMultiObs
import time


def draw_path(world, points, life_time=60.0):
    """在 CARLA 中绘制路径（绿色线）"""
    for i in range(len(points) - 1):
        world.debug.draw_line(
            points[i],
            points[i + 1],
            thickness=0.1,
            color=carla.Color(0, 255, 0),
            life_time=life_time
        )


def parse_targets(target_str):
    """解析目标点字符串: 'x1,y1;x2,y2;...' → [Location(...), ...]"""
    if not target_str:
        return []
    targets = []
    for pair in target_str.split(";"):
        x, y = map(float, pair.split(","))
        targets.append(carla.Location(x=x, y=y, z=0.0))
    return targets


def main():
    parser = argparse.ArgumentParser(description="增强版 CARLA 导航评估")
    parser.add_argument("--model_path", type=str, default="./checkpoints/best_model.zip")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--targets", type=str, default=None,
                        help='目标点序列，格式: "x1,y1;x2,y2;..."')
    parser.add_argument("--target_x", type=float, default=None)
    parser.add_argument("--target_y", type=float, default=None)
    parser.add_argument("--waypoint_dist", type=float, default=4.0)
    parser.add_argument("--steer_gain", type=float, default=1.8, help="转向增益")
    parser.add_argument("--arrival_radius", type=float, default=1.0, help="到达判定半径（米）")
    parser.add_argument("--visualize_path", action="store_true", help="在CARLA中绘制路径")
    args = parser.parse_args()

    # 解析目标点
    targets = parse_targets(args.targets)
    if args.target_x is not None and args.target_y is not None:
        targets.insert(0, carla.Location(x=args.target_x, y=args.target_y, z=0.0))

    print("🚀 启动增强版导航评估器...")
    print(f"🎯 目标点数量: {len(targets)}")
    if targets:
        for i, t in enumerate(targets):
            print(f"   {i + 1}. ({t.x:.1f}, {t.y:.1f})")

    env = None
    try:
        env = CarlaEnvMultiObs(keep_alive_after_exit=True, max_episode_steps=args.steps)
        model = PPO.load(args.model_path)
        obs, _ = env.reset()
        total_reward = 0.0
        current_target_idx = 0

        # 可视化路径
        if args.visualize_path and targets:
            draw_path(env.world, [env.vehicle.get_location()] + targets, life_time=120.0)

        print("\n▶️ 开始驾驶演示...\n")

        for step in range(args.steps):
            vehicle_tf = env.get_vehicle_transform()
            if vehicle_tf is None:
                print("⚠️ 车辆丢失，尝试重置...")
                obs, _ = env.reset()
                continue

            # 获取当前目标
            current_target = None
            if targets:
                if current_target_idx < len(targets):
                    current_target = targets[current_target_idx]
                    dist_to_target = vehicle_tf.location.distance(current_target)
                    if dist_to_target < args.arrival_radius:
                        print(f"✅ 到达第 {current_target_idx + 1} 个目标点！")
                        current_target_idx += 1
                        if current_target_idx >= len(targets):
                            print("🏁 所有目标点已到达！")
                            break
                else:
                    break  # 所有目标完成

            # 计算局部目标点
            if current_target:
                to_target = np.array([
                    current_target.x - vehicle_tf.location.x,
                    current_target.y - vehicle_tf.location.y
                ])
                direction = to_target / (np.linalg.norm(to_target) + 1e-6)
                local_target = carla.Location(
                    x=vehicle_tf.location.x + direction[0] * args.waypoint_dist,
                    y=vehicle_tf.location.y + direction[1] * args.waypoint_dist,
                    z=vehicle_tf.location.z
                )
            else:
                local_target = env.get_forward_waypoint(distance=args.waypoint_dist)

            # 计算转向
            steer = 0.0
            if local_target and vehicle_tf:
                forward = vehicle_tf.get_forward_vector()
                to_waypoint = np.array([
                    local_target.x - vehicle_tf.location.x,
                    local_target.y - vehicle_tf.location.y
                ])
                norm_fw = np.linalg.norm([forward.x, forward.y])
                norm_wp = np.linalg.norm(to_waypoint)
                if norm_fw > 1e-3 and norm_wp > 1e-3:
                    cos_angle = (forward.x * to_waypoint[0] + forward.y * to_waypoint[1]) / (norm_fw * norm_wp)
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                    cross = forward.x * to_waypoint[1] - forward.y * to_waypoint[0]
                    steer = np.clip(angle * np.sign(cross) * args.steer_gain, -1.0, 1.0)

            # 动态调速：弯道或接近目标时减速
            throttle_brake_action, _ = model.predict(obs, deterministic=True)
            throttle = float(np.clip(throttle_brake_action[0], 0.0, 1.0))
            brake = float(np.clip(throttle_brake_action[2], 0.0, 1.0))

            if current_target:
                dist = vehicle_tf.location.distance(current_target)
                if dist < 5.0:  # 接近目标
                    throttle *= (dist / 5.0)  # 线性减速
            if abs(steer) > 0.6:  # 急转弯
                throttle *= 0.7

            action = np.array([throttle, steer, brake])
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # 日志
            if step % 50 == 0:
                x, y, vx, vy = obs
                speed = np.linalg.norm([vx, vy])
                target_info = f" → 目标{current_target_idx + 1}" if current_target else ""
                print(f" Step {step:4d}: ({x:6.1f}, {y:6.1f}) @ {speed:4.1f} m/s{target_info}")

            if terminated or truncated:
                reason = "碰撞" if terminated else "超时"
                print(f"⏹️ 终止: {reason}")
                break

        print(f"\n✅ 演示结束 | 总奖励: {total_reward:.2f}")
        input("\n🛑 按 Enter 退出...")

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if env:
            env.close()


if __name__ == "__main__":
    main()
