"""
人形机器人强化学习训练 (优化版)
"""
import os
import time
import numpy as np
from datetime import datetime

from humanoid_rl_env import HumanoidRLEnv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor


class StatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.lengths = []
        self.start = time.time()

    def _on_step(self):
        if self.model.ep_info_buffer:
            for info in self.model.ep_info_buffer:
                self.rewards.append(info.get("r", 0))
                self.lengths.append(info.get("l", 0))

        if self.n_calls % 5000 == 0 and self.rewards:
            recent = min(100, len(self.rewards))
            mean_r = np.mean(self.rewards[-recent:])
            elapsed = time.time() - self.start
            print(f"Step {self.n_calls:8d} | Mean Reward: {mean_r:8.2f} | "
                  f"Episodes: {len(self.rewards):5d} | Time: {elapsed:.1f}s")

        return True


def train():
    print("=" * 60)
    print("       人形机器人强化学习训练 (力矩控制)")
    print("=" * 60)

    env = HumanoidRLEnv()
    env = Monitor(env)
    eval_env = HumanoidRLEnv(render_mode=None)

    # ====================== 修复：PPO 最优参数 ======================
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.97,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device="cpu",
    )

    callbacks = [
        StatsCallback(),
        EvalCallback(eval_env, best_model_save_path="./models/best_humanoid/",
                     eval_freq=10000, deterministic=True, render=False, verbose=1)
    ]

    print("\n开始训练 (2,000,000步)...")
    start = time.time()
    model.learn(total_timesteps=2_000_000, callback=callbacks)

    print(f"\n训练完成! 用时: {time.time()-start:.1f}秒")
    os.makedirs("./models", exist_ok=True)
    model.save("./models/humanoid_torque_final")
    print("模型已保存!")
    return model


def test(episodes=3):
    print("=" * 60)
    print("       测试模型")
    print("=" * 60)

    env = HumanoidRLEnv(render_mode="human")
    path = "./models/humanoid_torque_final.zip"
    if not os.path.exists(path):
        print("未找到模型，请先训练!")
        return

    model = PPO.load(path, env=env)
    print(f"已加载: {path}\n")

    for ep in range(episodes):
        obs, _ = env.reset()
        total_r = 0
        steps = 0
        done = False

        print(f"\n--- Episode {ep + 1}/{episodes} ---")
        while not done and steps < 2000:
            env.render()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_r += reward
            steps += 1

            if steps % 200 == 0:
                print(f"  Step {steps}: reward={total_r:.1f}")

        print(f"完成! 步数={steps}, 总奖励={total_r:.1f}")
    env.close()


def main():
    print("\n人形机器人 RL 训练")
    print("1. 训练新模型")
    print("2. 测试现有模型")
    print("3. 继续训练")

    choice = input("\n选择: ").strip()
    os.makedirs("./models", exist_ok=True)

    if choice == "1":
        train()
        if input("\n测试模型? [y/n]: ").strip().lower() == 'y':
            test()

    elif choice == "2":
        test()

    elif choice == "3":
        if os.path.exists("./models/humanoid_torque_final.zip"):
            env = HumanoidRLEnv()
            model = PPO.load("./models/humanoid_torque_final.zip", env=env)
            print("\n继续训练 1,000,000 步...")

            # 备份原模型（带时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"./models/humanoid_torque_backup_{timestamp}.zip"
            os.rename("./models/humanoid_torque_final.zip", backup_path)
            print(f"原模型已备份: {backup_path}")

            # 继续训练
            model.learn(total_timesteps=1_000_000, reset_num_timesteps=False)
            model.save("./models/humanoid_torque_final")
            print(f"训练完成! 模型已保存到: ./models/humanoid_torque_final.zip")
        else:
            print("未找到模型")

if __name__ == "__main__":
    main()