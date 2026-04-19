import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

import os
import gymnasium as gym
import torch
import numpy as np
from td3_agent import TD3Agent
from env_wrappers import wrap_env


def train():
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
        use_cnn=True,
        action_smooth_factor=0.3
    )

    max_episodes = 2000
    max_timesteps = 1000
    expl_noise = 0.15
    min_expl_noise = 0.02

    print("开始训练...")

    best_reward = -float('inf')
    episode_rewards = []

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        agent.reset_action_history()

        if episode < 300:
            agent.action_smoother.smooth_factor = 0.15
        elif episode < 800:
            agent.action_smoother.smooth_factor = 0.2
        else:
            agent.action_smoother.smooth_factor = 0.25

        action_stats = {'steering': [], 'gas': [], 'brake': []}

        for t in range(max_timesteps):
            action = agent.select_action(state, apply_smoothing=True)
            current_noise = expl_noise * (1 - episode / max_episodes * 0.9)
            noisy_action = action + np.random.normal(0, current_noise, size=action_dim)

            noisy_action[0] = np.clip(noisy_action[0], -max_action, max_action)
            noisy_action[1] = np.clip(noisy_action[1], 0, max_action)
            noisy_action[2] = np.clip(noisy_action[2], 0, max_action)

            if noisy_action[1] > 0.3 and noisy_action[2] > 0.3:
                noisy_action[1] = 0.1

            action_stats['steering'].append(noisy_action[0])
            action_stats['gas'].append(noisy_action[1])
            action_stats['brake'].append(noisy_action[2])

            next_state, reward, terminated, truncated, _ = env.step(noisy_action)
            done = terminated or truncated

            forward_reward = noisy_action[1] * 0.5
            brake_penalty = -noisy_action[2] * 0.3
            steering_penalty = -abs(noisy_action[0]) * 0.05
            shaped_reward = reward + forward_reward + brake_penalty + steering_penalty

            if reward < -0.1:
                shaped_reward -= 0.2

            agent.replay_buffer.add((state, noisy_action, shaped_reward, next_state, done))
            state = next_state
            episode_reward += reward
            agent.train()

            if done:
                break

        expl_noise = max(min_expl_noise, expl_noise * 0.998)

        avg_steering = np.mean(action_stats['steering']) if action_stats['steering'] else 0
        avg_gas = np.mean(action_stats['gas']) if action_stats['gas'] else 0
        avg_brake = np.mean(action_stats['brake']) if action_stats['brake'] else 0

        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else episode_reward

        print(
            f"回合: {episode + 1:4d} | 奖励: {episode_reward:6.1f} | 平均(50): {avg_reward:6.1f} | 噪声: {current_noise:.3f} | 平滑: {agent.action_smoother.smooth_factor:.2f}")
        print(f"  动作 - 转向: {avg_steering:+.2f}, 油门: {avg_gas:.2f}, 刹车: {avg_brake:.2f}")

        if episode_reward > best_reward and episode > 100:
            best_reward = episode_reward
            os.makedirs("models", exist_ok=True)
            agent.save(f"models/td3_best")
            print(f"  ★ 新最佳模型! 奖励: {best_reward:.1f}")

        if (episode + 1) % 50 == 0:
            os.makedirs("models", exist_ok=True)
            agent.save(f"models/td3_car_{episode + 1}")
            print(f"  模型已保存: models/td3_car_{episode + 1}")

    env.close()
    print("训练完成!")


if __name__ == "__main__":
    train()