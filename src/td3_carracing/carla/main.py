import os
import time
import torch
import numpy as np
from carla_env import CarlaEnv
from td3_agent import TD3Agent
from env_wrappers import wrap_env


def train():
    # 创建环境
    env = CarlaEnv(town="Town03", render_mode="human", max_episode_steps=1000)
    env = wrap_env(env)

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    # 动作空间范围：[steer, throttle, brake]
    max_action = np.array([1.0, 1.0, 1.0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        use_cnn=True
    )

    max_episodes = 2000
    max_timesteps = 1000
    warmup_episodes = 50

    # 探索噪声设置
    expl_noise_steer = 0.1
    expl_noise_throttle = 0.15
    expl_noise_brake = 0.08
    min_noise_steer = 0.005
    min_noise_throttle = 0.02
    min_noise_brake = 0.01

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
        collision_counter = 0
        lane_invasion_counter = 0
        on_track_counter = 0

        for t in range(max_timesteps):
            # 预热期：使用更保守的策略
            if episode < warmup_episodes:
                # 预热阶段：降低探索，更多利用
                action = agent.select_action(state, smooth=True)
                noise = np.zeros_like(action)
                noise[0] = np.random.normal(0, expl_noise_steer * 0.3)
                noise[1] = np.random.normal(0, expl_noise_throttle * 0.5)
                noise[2] = np.random.normal(0, expl_noise_brake * 0.3)
            else:
                action = agent.select_action(state, smooth=True)
                noise = np.zeros_like(action)
                noise[0] = np.random.normal(0, expl_noise_steer)
                noise[1] = np.random.normal(0, expl_noise_throttle)
                noise[2] = np.random.normal(0, expl_noise_brake)

            # 如果刚发生碰撞，降低转向探索，增加刹车
            if collision_counter > 0 and collision_counter < 5:
                noise[0] *= 0.3
                noise[1] *= 0.5  # 减少油门
                noise[2] *= 1.2  # 稍微增加刹车

            action = (action + noise).clip(-1.0, 1.0)
            # 确保油门和刹车在有效范围
            action[1] = np.clip(action[1], 0.0, 1.0)
            action[2] = np.clip(action[2], 0.0, 1.0)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 更新状态计数
            collision = info.get('collision', False)
            lane_invasion = info.get('lane_invasion', False)
            speed = info.get('speed', 0.0)

            if collision:
                collision_counter += 1
            else:
                collision_counter = 0

            if lane_invasion:
                lane_invasion_counter += 1
            else:
                lane_invasion_counter = 0
                on_track_counter += 1
                if on_track_counter > 3:
                    collision_counter = 0

            # 检测车辆打转
            if abs(speed) < 1.0 and abs(action[0]) > 0.5:
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
            agent.save(f"models/td3_carla_best")
            print(f"★ 新最佳模型！奖励: {episode_reward:.1f}")

        # 动态噪声衰减
        if episode > warmup_episodes:
            expl_noise_steer = max(min_noise_steer, expl_noise_steer * 0.998)
            expl_noise_throttle = max(min_noise_throttle, expl_noise_throttle * 0.999)
            expl_noise_brake = max(min_noise_brake, expl_noise_brake * 0.999)

        avg_reward = np.mean(reward_history[-10:]) if len(reward_history) >= 10 else episode_reward

        print(
            f"回合: {episode + 1}, 奖励: {episode_reward:.1f}, 平均奖励(10): {avg_reward:.1f}, "
            f"转向噪声: {expl_noise_steer:.4f}, 油门噪声: {expl_noise_throttle:.3f}, "
            f"刹车噪声: {expl_noise_brake:.3f}, 缓冲区: {len(agent.replay_buffer)}"
        )

        if (episode + 1) % 100 == 0:
            os.makedirs("models", exist_ok=True)
            agent.save(f"models/td3_carla_{episode + 1}")
            print(f"模型已保存: models/td3_carla_{episode + 1}")

    env.close()
    print("训练完成！")


def test():
    """测试训练好的模型"""
    # 创建环境
    env = CarlaEnv(town="Town03", render_mode="human", max_episode_steps=1000)
    env = wrap_env(env)

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = np.array([1.0, 1.0, 1.0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        use_cnn=True
    )

    # 加载模型
    try:
        agent.load("models/td3_carla_best")
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    max_episodes = 10

    print("开始测试...")
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        agent.last_action = None

        for t in range(1000):
            action = agent.select_action(state, smooth=True, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward

            if done:
                break

            # 稍微延迟以便观察
            time.sleep(0.01)

        print(f"测试回合: {episode + 1}, 奖励: {episode_reward:.1f}")

    env.close()
    print("测试完成！")


if __name__ == "__main__":
    # 检查是否有训练好的模型
    if os.path.exists("models/td3_carla_best_actor.pth") and os.path.exists("models/td3_carla_best_critic1.pth"):
        print("发现训练好的模型，开始测试...")
        test()
    else:
        print("未发现训练好的模型，开始训练...")
        train()
