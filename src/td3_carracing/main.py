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
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        use_cnn=True
    )

    max_episodes = 1000
    max_timesteps = 1000
    expl_noise = 0.1

    print("开始训练...")
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for t in range(max_timesteps):
            action = agent.select_action(state)
            action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(-max_action, max_action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.add((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward

            agent.train()

            if done:
                break

        print(f"回合: {episode + 1}, 奖励: {episode_reward:.1f}, 噪声: {expl_noise:.2f}")
        expl_noise = max(0.01, expl_noise * 0.995)

        if (episode + 1) % 50 == 0:
            os.makedirs("models", exist_ok=True)
            agent.save(f"models/td3_car_{episode + 1}")
            print(f"模型已保存: models/td3_car_{episode + 1}")

    env.close()

if __name__ == "__main__":
    train()