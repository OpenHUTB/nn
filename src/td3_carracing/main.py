import os
# 显示游戏窗口（注释掉则使用虚拟显示）
# os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
import torch
import numpy as np
from td3_agent import TD3Agent
from env_wrappers import wrap_env


def train():
    # 创建环境（显示窗口）
    env = gym.make("CarRacing-v3", render_mode="human")
    env = wrap_env(env)

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        use_cnn=True
    )

    max_episodes = 1000
    max_timesteps = 1000
    # 差异化探索噪声：转向噪声更低，油门/刹车噪声正常
    expl_noise_steer = 0.02  # 转向探索噪声（原0.05）
    expl_noise_throttle = 0.05  # 油门/刹车噪声
    best_reward = -float('inf')

    print("开始训练...")
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        agent.last_action = None  # 重置动作历史

        for t in range(max_timesteps):
            action = agent.select_action(state, smooth=True)  # 启用平滑

            # 差异化添加探索噪声：转向噪声单独降低
            noise = np.zeros_like(action)
            noise[0] = np.random.normal(0, expl_noise_steer)  # 转向噪声
            noise[1:] = np.random.normal(0, expl_noise_throttle, size=2)  # 油门/刹车噪声
            action = (action + noise).clip(-max_action, max_action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.add((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward

            agent.train()

            if done:
                break

        # 保存最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            os.makedirs("models", exist_ok=True)
            agent.save(f"models/td3_car_best")
            print(f"★ 新最佳模型！奖励: {episode_reward:.1f}")

        # 噪声衰减：转向噪声衰减更快
        expl_noise_steer = max(0.005, expl_noise_steer * 0.995)
        expl_noise_throttle = max(0.02, expl_noise_throttle * 0.998)

        print(
            f"回合: {episode + 1}, 奖励: {episode_reward:.1f}, 转向噪声: {expl_noise_steer:.4f}, "
            f"油门噪声: {expl_noise_throttle:.3f}, 缓冲区: {len(agent.replay_buffer)}")

        if (episode + 1) % 50 == 0:
            os.makedirs("models", exist_ok=True)
            agent.save(f"models/td3_car_{episode + 1}")
            print(f"模型已保存: models/td3_car_{episode + 1}")

    env.close()
    print("训练完成！")


if __name__ == "__main__":
    train()