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

    max_episodes = 2500
    max_timesteps = 1200
    warmup_episodes = 80  # 增加预热回合数，让策略更好地收敛

    # 探索噪声设置 - 优化转向探索
    expl_noise_steer = 0.12  # 稍微增加转向探索
    expl_noise_throttle = 0.12  # 降低油门探索
    expl_noise_brake = 0.05
    min_noise_steer = 0.01
    min_noise_throttle = 0.02
    min_noise_brake = 0.005

    best_reward = -float('inf')
    reward_history = []

    print("开始训练...")
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        agent.last_action = None

        spin_counter = 0

        # 学习率调整
        if episode % 500 == 0 and episode > 0:
            for param_group in agent.actor_optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.8
            for param_group in agent.critic_optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.8

        # 记录出赛道状态
        off_track_counter = 0
        on_track_counter = 0

        for t in range(max_timesteps):
            # 预热期：增加转向探索，让智能体学习转弯
            if episode < warmup_episodes:
                action = agent.select_action(state, smooth=True)
                noise = np.zeros_like(action)
                noise[0] = np.random.normal(0, expl_noise_steer * 0.6)
                noise[1] = np.random.normal(0, expl_noise_throttle * 0.5)
                noise[2] = np.random.normal(0, expl_noise_brake * 0.3)
            else:
                action = agent.select_action(state, smooth=True)
                noise = np.zeros_like(action)
                noise[0] = np.random.normal(0, expl_noise_steer)
                noise[1] = np.random.normal(0, expl_noise_throttle)
                noise[2] = np.random.normal(0, expl_noise_brake)

            # 如果刚出赛道，降低转向探索
            if off_track_counter > 0 and off_track_counter < 5:
                noise[0] *= 0.3
                noise[1] *= 1.2  # 稍微增加油门，鼓励回到赛道

            action = (action + noise).clip(-max_action, max_action)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 更新赛道状态计数
            on_track = info.get('on_track', True)
            if not on_track:
                off_track_counter += 1
                on_track_counter = 0
            else:
                on_track_counter += 1
                if on_track_counter > 3:
                    off_track_counter = 0

            speed = info.get('speed', 0.0)
            if abs(speed) < 0.2 and abs(action[0]) > 0.4:
                spin_counter += 1
                reward -= 0.3
                if spin_counter > 15:
                    reward -= 5.0
                    truncated = True
            else:
                spin_counter = 0

            agent.replay_buffer.add((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward

            agent.train()

            if done:
                break

        reward_history.append(episode_reward)

        if episode_reward > best_reward:
            best_reward = episode_reward
            os.makedirs("models", exist_ok=True)
            agent.save(f"models/td3_car_best")
            print(f"★ 新最佳模型！奖励: {episode_reward:.1f}")

        # 动态噪声衰减
        if episode > warmup_episodes:
            expl_noise_steer = max(min_noise_steer, expl_noise_steer * 0.999)
            expl_noise_throttle = max(min_noise_throttle, expl_noise_throttle * 0.9995)
            expl_noise_brake = max(min_noise_brake, expl_noise_brake * 0.9995)

        avg_reward = np.mean(reward_history[-10:]) if len(reward_history) >= 10 else episode_reward

        print(
            f"回合: {episode + 1}, 奖励: {episode_reward:.1f}, 平均奖励(10): {avg_reward:.1f}, "
            f"转向噪声: {expl_noise_steer:.4f}, 油门噪声: {expl_noise_throttle:.3f}, "
            f"缓冲区: {len(agent.replay_buffer)}")

        if (episode + 1) % 100 == 0:
            os.makedirs("models", exist_ok=True)
            agent.save(f"models/td3_car_{episode + 1}")
            print(f"模型已保存: models/td3_car_{episode + 1}")

    env.close()
    print("训练完成！")


if __name__ == "__main__":
    train()