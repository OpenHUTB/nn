"""
================================================================================
Double DQN 训练脚本
================================================================================
训练 Double DQN 智能体玩 CarRacing-v3 游戏。

Double DQN 相对于 DQN 的改进:
================================================================================
1. Double DQN: 解决 Q 值过估计问题
2. Soft Update: 更平滑的目标网络更新
3. 梯度裁剪: 防止梯度爆炸
4. 学习率调度: 自动调整学习率

与 DQN 训练脚本的主要区别:
- 配置文件不同
- 使用 DoubleDQNAgent
- 支持学习率调度器和梯度裁剪
================================================================================
"""
import os
import sys
import matplotlib
import torch
import datetime
import csv
from pathlib import Path

import gymnasium as gym
import gymnasium.wrappers as gym_wrap
import matplotlib.pyplot as plt
import numpy as np

# 添加路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入 DoubleDQN 智能体（使用新模块）
from doubledqn_agent import DoubleDQNAgent as Agent, SkipFrame, plot_reward

from gymnasium.spaces import Box
from tensordict import TensorDict
from torch import nn
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# 开启交互式绘图
plt.ion()


# ================================================================================
# 1. 环境设置
# ================================================================================
print("=" * 50)
print("正在初始化环境...")
print("=" * 50)

# 创建环境
env = gym.make("CarRacing-v3", continuous=False)

# 预处理
env = SkipFrame(env, skip=4)

from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation

env = GrayscaleObservation(env)
env = ResizeObservation(env, (84, 84))
env = FrameStackObservation(env, stack_size=4)

# 重置环境
state, info = env.reset()
action_n = env.action_space.n

print(f"动作空间大小: {action_n}")
print(f"状态空间形状: {state.shape}")
print("环境初始化完成！")
print("=" * 50)


# ================================================================================
# 2. 智能体初始化
# ================================================================================
print("\n正在创建 DoubleDQN 智能体...")

config_path = Path(__file__).parent.parent / 'configs' / 'double_dqn.yaml'

driver = Agent(
    state_space_shape=state.shape,
    action_n=action_n,
    config_path=config_path,
    load_state=False,
    load_model=None
)

print(f"使用设备: {driver.device}")
print(f"折扣因子 (gamma): {driver.gamma}")
print(f"软更新系数 (tau): {driver.tau}")
print(f"目标网络更新间隔: {driver.update_target_every}")
print(f"初始探索率 (epsilon): {driver.epsilon}")
print(f"学习率: {driver.hyperparameters.get('lr', 0.0001)}")

if driver.scheduler:
    print(f"学习率调度: StepLR (step={driver.scheduler.step_size}, gamma={driver.scheduler.gamma})")

print("智能体创建完成！")


# ================================================================================
# 3. 训练参数
# ================================================================================
batch_n = 32
play_n_episodes = 2000

episode_reward_list = []
episode_length_list = []
episode_loss_list = []
episode_epsilon_list = []
episode_date_list = []
episode_time_list = []

episode = 0
timestep_n = 0
when2learn = 4
when2log = 10

report_type = 'plot'


# ================================================================================
# 4. 训练主循环
# ================================================================================
print("\n" + "=" * 50)
print("开始训练 DoubleDQN！")
print("=" * 50)

while episode < play_n_episodes:
    episode += 1
    episode_reward = 0
    episode_length = 0
    loss_list = []
    episode_epsilon_list.append(driver.epsilon)
    
    state, info = env.reset()
    
    done = False
    while not done:
        timestep_n += 1
        episode_length += 1
        
        # 选择动作
        action = driver.take_action(state)
        
        # 与环境交互
        new_state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        # 存储经验
        driver.store(state, action, reward, new_state, terminated)
        
        state = new_state
        done = terminated or truncated
        
        # 训练网络
        if timestep_n % when2learn == 0 and len(driver.buffer) >= batch_n:
            q_value, loss = driver.update_net(batch_n)
            loss_list.append(loss)
        
        # 打印训练进度
        if report_type == 'text' and timestep_n % 5000 == 0:
            print(f"\n[t={timestep_n}] Episode {episode}")
            print(f"    epsilon: {driver.epsilon:.4f}")
            print(f"    n_updates: {driver.n_updates}")
            if driver.scheduler:
                print(f"    learning rate: {driver.get_current_lr():.6f}")
    
    # 记录结果
    episode_reward_list.append(episode_reward)
    episode_length_list.append(episode_length)
    episode_loss_list.append(np.mean(loss_list) if loss_list else 0)
    
    now_time = datetime.datetime.now()
    episode_date_list.append(now_time.date().isoformat())
    episode_time_list.append(now_time.time().isoformat())
    
    # 绘图
    if report_type == 'plot':
        plot_reward(episode, episode_reward_list, timestep_n)
    
    # 保存日志
    if episode % when2log == 0:
        driver.write_log(
            episode_date_list,
            episode_time_list,
            episode_reward_list,
            episode_length_list,
            episode_loss_list,
            episode_epsilon_list,
            log_filename='DoubleDQN_log.csv'
        )
    
    # 打印结果
    if episode % 10 == 0:
        recent_rewards = episode_reward_list[-10:]
        mean_reward = np.mean(recent_rewards)
        lr_info = f", LR: {driver.get_current_lr():.6f}" if driver.scheduler else ""
        print(f"Episode {episode}/{play_n_episodes} | "
              f"Reward: {episode_reward:.1f} | "
              f"Mean(10): {mean_reward:.1f} | "
              f"Steps: {episode_length} | "
              f"Epsilon: {driver.epsilon:.4f}"
              f"{lr_info}")


# ================================================================================
# 5. 评估
# ================================================================================
print("\n" + "=" * 50)
print("训练完成！开始评估...")
print("=" * 50)

def evaluate_agent(agent, num_episodes=5, render=False):
    """
    评估训练好的智能体
    
    评估时:
    - epsilon = 0: 完全利用
    - 使用固定种子
    - 计算平均得分
    """
    render_mode = "human" if render else "rgb_array"
    eval_env = gym.make("CarRacing-v3", continuous=False, render_mode=render_mode)
    eval_env = SkipFrame(eval_env, skip=4)
    eval_env = GrayscaleObservation(eval_env)
    eval_env = ResizeObservation(eval_env, (84, 84))
    eval_env = FrameStackObservation(eval_env, stack_size=4)
    
    agent.epsilon = 0
    
    scores = []
    for ep in range(num_episodes):
        state, _ = eval_env.reset(seed=ep)
        score = 0
        done = False
        
        while not done:
            action = agent.take_action(state)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            score += reward
            done = terminated or truncated
        
        scores.append(score)
        print(f"评估 Episode {ep+1}/{num_episodes} | 种子: {ep} | 得分: {score:.1f}")
    
    eval_env.close()
    return np.mean(scores)


avg_score = evaluate_agent(driver, num_episodes=5, render=True)

print("=" * 50)
print(f"评估完成！平均得分: {avg_score:.1f}")
if avg_score >= 900:
    print("🎉 优秀！DoubleDQN 表现优异！")
elif avg_score >= 700:
    print("👍 良好！")
elif avg_score >= 400:
    print("📈 一般")
else:
    print("⚠️ 建议继续训练")
print("=" * 50)


# ================================================================================
# 6. 保存
# ================================================================================
print("\n正在保存最终模型...")
driver.save(driver.save_dir, 'DoubleDQN_final')
driver.write_log(
    episode_date_list,
    episode_time_list,
    episode_reward_list,
    episode_length_list,
    episode_loss_list,
    episode_epsilon_list,
    log_filename='DoubleDQN_log.csv'
)

env.close()
plt.ioff()
print("\n训练脚本执行完毕！")
