# eval_agent.py
"""
评估已训练的 PPO 自动驾驶智能体
支持两种模式：
1. 默认：沿车道中心自动前进（使用 get_forward_waypoint）
2. 指定目标点：导航到 (target_x, target_y)
"""

import argparse
import numpy as np
import carla
from stable_baselines3 import PPO
from carla_env.carla_env_multi_obs import CarlaEnvMultiObs


def main():
    parser = argparse.ArgumentParser(description="评估 CARLA PPO 自动驾驶智能体")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./checkpoints/best_model.zip",
        help="已训练模型路径（.zip 文件）"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="最大运行步数"
    )
    parser.add_argument(
        "--target_x",
        type=float,
        default=None,
        help="全局目标点 x 坐标（世界坐标系，单位：米）"
    )
    parser.add_argument(
        "--target_y",
        type=float,
        default=None,
        help="全局目标点 y 坐标（世界坐标系，单位：米）"
    )
    parser.add_argument(
        "--waypoint_dist",
        type=float,
        default=4.0,
        help="局部目标点前瞻距离（米），建议 2.0~5.0"
    )
    args = parser.parse_args()

    print("🚀 正在启动评估环境...")
    print("💡 请确保 CARLA 仿真器（CarlaUE4.exe）已在后台运行！\n")

    try:
        # 创建环境（保留车辆以便观察）
        env = CarlaEnvMultiObs(keep_alive_after_exit=True, max_episode_steps=args.steps)

        # 加载模型（仅用于底层控制：油门/刹车）
        print(f"📂 加载模型: {args.model_path}")
        model = PPO.load(args.model_path)

        # 初始化环境
        print("🔄 重置环境并生成车辆...")
        obs, _ = env.reset()
        total_reward = 0.0

        # 设置全局目标点（如果提供）
        global_target = None
        if args.target_x is not None and args.target_y is not None:
            global_target = carla.Location(x=args.target_x, y=args.target_y, z=0.0)
            print(f"🎯 全局目标点: ({args.target_x:.1f}, {args.target_y:.1f})")
        else:
            print("🛣️ 未指定目标点，将沿车道自动前进...")

        print("\n▶️ 开始驾驶演示...\n")

        for step in range(args.steps):
            # ===== 高层导航逻辑：计算局部目标点 =====
            local_target = None
            vehicle_tf = env.get_vehicle_transform()

            if vehicle_tf is None:
                print("⚠️ 车辆状态异常，终止演示")
                break

            if global_target is not None:
                # --- 模式1：朝向全局目标点 ---
                to_target = np.array([
                    global_target.x - vehicle_tf.location.x,
                    global_target.y - vehicle_tf.location.y
                ])
                dist_to_target = np.linalg.norm(to_target)

                if dist_to_target < 1.0:
                    print("🏁 已到达目标点！")
                    break

                # 计算单位方向向量
                direction = to_target / (dist_to_target + 1e-6)
                local_target = carla.Location(
                    x=vehicle_tf.location.x + direction[0] * args.waypoint_dist,
                    y=vehicle_tf.location.y + direction[1] * args.waypoint_dist,
                    z=vehicle_tf.location.z
                )
            else:
                # --- 模式2：沿车道中心前进 ---
                local_target = env.get_forward_waypoint(distance=args.waypoint_dist)
                if local_target is None:
                    print("⚠️ 无法获取前方路点，使用原始策略")
                    local_target = None

            # ===== 底层控制：结合 PPO 与转向决策 =====
            if local_target is not None:
                # 计算期望转向角（基于局部目标）
                forward = vehicle_tf.get_forward_vector()
                to_waypoint = np.array([
                    local_target.x - vehicle_tf.location.x,
                    local_target.y - vehicle_tf.location.y
                ])

                # 避免除零
                norm_fw = np.linalg.norm([forward.x, forward.y])
                norm_wp = np.linalg.norm(to_waypoint)
                if norm_fw < 1e-3 or norm_wp < 1e-3:
                    steer = 0.0
                else:
                    # 计算夹角（使用叉积判断左右）
                    cos_angle = (forward.x * to_waypoint[0] + forward.y * to_waypoint[1]) / (norm_fw * norm_wp)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)  # [0, π]

                    # 叉积符号决定转向方向
                    cross = forward.x * to_waypoint[1] - forward.y * to_waypoint[0]
                    steer = np.clip(angle * np.sign(cross) * 1.5, -1.0, 1.0)  # 比例增益可调

                # 使用 PPO 决定油门和刹车（输入仍为原始 4D 观测）
                throttle_brake_action, _ = model.predict(obs, deterministic=True)
                throttle = float(np.clip(throttle_brake_action[0], 0.0, 1.0))
                brake = float(np.clip(throttle_brake_action[2], 0.0, 1.0))

                action = np.array([throttle, steer, brake])
            else:
                # 回退到纯 PPO 策略
                action, _ = model.predict(obs, deterministic=True)

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # 定期打印状态
            if step % 50 == 0 or step == args.steps - 1:
                x, y, vx, vy = obs
                speed = np.linalg.norm([vx, vy])
                print(f" Step {step:3d}: 位置=({x:6.1f}, {y:6.1f}), 速度={speed:5.2f} m/s")

            # 终止条件
            if terminated or truncated:
                reason = "碰撞" if terminated else "超时"
                print(f"⏹️ 演示结束（原因: {reason}）")
                break

        print(f"\n✅ 演示完成！总奖励: {total_reward:.2f}")
        print("ℹ️ 车辆已保留在 CARLA 中，可自由观察。")
        input("\n🛑 按 Enter 键退出并销毁车辆...")

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            env.close()
        except:
            pass


if __name__ == "__main__":
    main()
