import os
# 显示游戏窗口
# os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
import torch
import numpy as np
from td3_agent import TD3Agent
from env_wrappers import wrap_env


def train():
    # 显示游戏窗口
    env = gym.make("CarRacing-v3", render_mode="human")
    env = wrap_env(env)

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建带有动作平滑的代理
    # action_smooth_factor: 0.1-0.5之间，越小越平滑但响应越慢
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        use_cnn=True,
        action_smooth_factor=0.3  # 添加动作平滑
    )

    max_episodes = 1000
    max_timesteps = 1000
    expl_noise = 0.15  # 稍微降低初始噪声
    min_expl_noise = 0.02  # 最小探索噪声

    print("开始训练...")

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0

        # 重置动作历史（每个episode开始）
        agent.reset_action_history()

        # 动态调整平滑因子（初期更平滑，后期更响应）
        if episode < 200:
            agent.action_smoother.smooth_factor = 0.4  # 更平滑
        elif episode < 500:
            agent.action_smoother.smooth_factor = 0.3
        else:
            agent.action_smoother.smooth_factor = 0.2  # 更响应

        for t in range(max_timesteps):
            # 选择平滑后的动作
            action = agent.select_action(state, apply_smoothing=True)

            # 添加随时间衰减的探索噪声
            current_noise = expl_noise * (1 - episode / max_episodes * 0.8)
            noisy_action = (action + np.random.normal(0, current_noise, size=action_dim))
            noisy_action = np.clip(noisy_action, -max_action, max_action)

            next_state, reward, terminated, truncated, _ = env.step(noisy_action)
            done = terminated or truncated

            # 奖励整形：鼓励平滑动作
            if t > 0:
                action_change = np.abs(noisy_action - agent.action_history[-2] if len(agent.action_history) > 1 else 0)
                smoothness_bonus = -0.01 * np.mean(action_change)  # 惩罚剧烈的动作变化
                reward += smoothness_bonus

            agent.replay_buffer.add((state, noisy_action, reward, next_state, done))
            state = next_state
            episode_reward += reward

            agent.train()

            if done:
                break

        # 衰减探索噪声
        expl_noise = max(min_expl_noise, expl_noise * 0.998)

        print(f"回合: {episode + 1}, 奖励: {episode_reward:.1f}, "
              f"噪声: {current_noise:.3f}, 平滑系数: {agent.action_smoother.smooth_factor:.2f}")

        if (episode + 1) % 50 == 0:
            os.makedirs("models", exist_ok=True)
            agent.save(f"models/td3_car_{episode + 1}")
            print(f"模型已保存: models/td3_car_{episode + 1}")

    env.close()


if __name__ == "__main__":
    train()