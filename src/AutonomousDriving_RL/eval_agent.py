# eval_agent.py
"""
加载训练好的模型，在 CARLA 中演示智能体驾驶行为
"""

import argparse
import numpy as np
from stable_baselines3 import PPO
from carla_env.carla_env_multi_obs import CarlaEnvMultiObs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoints/best_model.zip", help="模型路径")
    parser.add_argument("--steps", type=int, default=200, help="演示步数")
    args = parser.parse_args()

    print("🔄 加载环境与模型...")
    env = CarlaEnvMultiObs(keep_alive_after_exit=True)  # 保留车辆便于观察
    model = PPO.load(args.model_path, env=env)

    print("▶️ 开始驾驶演示（运行 {} 步）...".format(args.steps))
    obs, _ = env.reset()
    total_reward = 0.0

    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # 每 50 步打印状态
        if step % 50 == 0 or step == args.steps - 1:
            x, y, vx, vy = obs
            speed = np.linalg.norm([vx, vy])
            print(f" Step {step}: 位置=({x:.1f}, {y:.1f}), 速度={speed:.2f} m/s")

        if terminated or truncated:
            break

    print(f"✅ 演示完成！总奖励: {total_reward:.2f}")
    input("🛑 准备好后，请回到本窗口按 Enter 键退出...")
    env.close()


if __name__ == "__main__":
    main()