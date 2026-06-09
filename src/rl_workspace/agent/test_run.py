import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import sys
sys.path.insert(0, 'src/examples')
import QLearner as ql

print("=" * 60)
print("FrozenLake 强化学习 Q-Learning 算法演示")
print("=" * 60)

env = gym.make(
    "FrozenLake-v1",
    desc=generate_random_map(size=4),
    is_slippery=False,
    render_mode="ansi",
    max_episode_steps=1000,
)

print(f"\n环境信息:")
print(f"  地图大小: 4x4")
print(f"  状态空间: {env.observation_space.n} (共 {env.observation_space.n} 个状态)")
print(f"  动作空间: {env.action_space.n} (上下左右)")
print(f"\n开始训练 Q-Learning 代理...")

EPOCHS = 1000
learner = ql.QLearner(
    states=env.observation_space.n,
    actions=env.action_space.n,
    radr=0.001
)

rewards = []
log_interval = 100

for episode in range(EPOCHS):
    state = env.reset()[0]
    done = False
    total_rewards = 0
    action = learner.get_next_action_without_Q_table_update(state)

    while not done:
        new_state, reward, done, trunc, info = env.step(action)
        action = learner.get_next_action_with_Q_table_update(new_state, reward)
        total_rewards += reward

    learner.decay_rar(episode)
    rewards.append(total_rewards)

    if episode % log_interval == 0:
        avg_reward = np.mean(rewards[-log_interval:]) * 100
        success_rate = np.sum([r > 0 for r in rewards[-log_interval:]]) / log_interval * 100
        print(f"  Episode {episode:4d}/{EPOCHS}: "
              f"平均奖励 = {avg_reward:6.2f}%, "
              f"成功率 = {success_rate:6.1f}%")

env.close()

print("\n" + "=" * 60)
print("训练完成!")
print(f"最终100轮平均奖励: {np.mean(rewards[-100:]) * 100:.2f}%")
print(f"成功到达目标的次数: {np.sum([r > 0 for r in rewards])}/{len(rewards)}")
print("=" * 60)

np.save("test_frozen_lake_q_table.npy", learner.Q)
print("\nQ表已保存到: test_frozen_lake_q_table.npy")
