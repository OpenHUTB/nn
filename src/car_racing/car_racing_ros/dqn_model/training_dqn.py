"""
================================================================================
DQN 训练脚本
================================================================================
训练 DQN 智能体玩 CarRacing-v3 游戏。

训练流程:
================================================================================
1. 环境预处理
   - CarRacing-v3 输出 RGB 图像 (96, 96, 3)
   - 灰度化: 减少计算量
   - 缩放到 (84, 84): 适配网络输入
   - 帧堆叠 (4帧): 提供时序信息，让智能体感知速度
   
2. 主循环
   - 探索环境，收集经验
   - 存储经验到回放缓冲区
   - 定期从缓冲区采样训练网络
   
3. 评估
   - 训练结束后评估智能体性能
================================================================================
"""
import os
import sys
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import torch
import datetime
import csv

import gymnasium as gym
import gymnasium.wrappers as gym_wrap
import matplotlib.pyplot as plt
import numpy as np

# 添加路径以导入自定义模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入 DQN 模型（现在使用新的 dqn_agent 模块）
# 旧版本: import DQN_model as DQN
# 新版本: from dqn_agent import DQNAgent
from dqn_agent import DQNAgent as Agent, SkipFrame, plot_reward

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

# 创建 CarRacing 环境
# continuous=False: 使用离散动作空间 (5个动作: 左/右/加速/刹车/空)
env = gym.make("CarRacing-v2", continuous=False)

# 应用环境预处理 wrapper
env = SkipFrame(env, skip=4)  # 每4步执行一次相同动作

from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

# 灰度化处理: RGB -> 灰度，减少信息量加速训练
env = GrayScaleObservation(env)

# 缩放到 (84, 84): 标准 Atari 游戏预处理尺寸
env = ResizeObservation(env, (84, 84))

# 帧堆叠: 将连续4帧组合在一起，提供时序信息
# 这样网络可以看到小车的运动方向
env = FrameStack(env, num_stack=4)

# 重置环境，获取初始状态
state, info = env.reset()

# 获取动作空间大小
action_n = env.action_space.n
print(f"动作空间大小: {action_n}")
print(f"状态空间形状: {state.shape}")  # 应该是 (4, 84, 84)
print("环境初始化完成！")
print("=" * 50)


# ================================================================================
# 2. 智能体初始化
# ================================================================================
print("\n正在创建智能体...")

driver = Agent(
    state_space_shape=state.shape,
    action_n=action_n,
    config_path='configs/dqn.yaml',
)

print(f"使用设备: {driver.device}")
print(f"折扣因子 (gamma): {driver.gamma}")
print(f"初始探索率 (epsilon): {driver.epsilon}")
print(f"探索率衰减: {driver.epsilon_decay}")
print(f"最小探索率: {driver.epsilon_min}")
print("智能体创建完成！")


# ================================================================================
# 3. 训练参数设置
# ================================================================================

# 批大小: 每次从回放缓冲区采样32条经验
batch_n = 32

# 训练 episode 总数
play_n_episodes = 2000

# 用于记录训练过程的列表
episode_reward_list = []      # 每个 episode 的总奖励
episode_length_list = []      # 每个 episode 的步数
episode_loss_list = []        # 每个 episode 的平均损失
episode_epsilon_list = []     # 每个 episode 结束时的探索率
episode_date_list = []        # 日期
episode_time_list = []        # 时间

# 训练统计
episode = 0                   # 当前 episode 编号
timestep_n = 0               # 总步数

# 控制训练频率的参数
when2learn = 4               # 每隔几步学习一次
when2sync = 5000             # 每隔几步同步目标网络
when2save = 100000           # 每隔几步保存模型
when2log = 10                # 每隔几个 episode 写入日志

# 报告类型: 'plot' 显示实时曲线, 'text' 打印文字, None 静默
report_type = 'text'


# ================================================================================
# 4. 训练主循环
# ================================================================================
print("\n" + "=" * 50)
print("开始训练！")
print("=" * 50)

while episode < play_n_episodes:
    episode += 1
    episode_reward = 0        # 当前 episode 的累积奖励
    episode_length = 0        # 当前 episode 的步数
    loss_list = []           # 当前 episode 的损失列表
    episode_epsilon_list.append(driver.epsilon)
    
    # 重置环境，开始新的 episode
    state, info = env.reset()
    
    # episode 主循环
    done = False
    while not done:
        timestep_n += 1
        episode_length += 1
        
        # -------------------------------------------------
        # 4.1 选择动作
        # -------------------------------------------------
        action = driver.take_action(state)
        
        # -------------------------------------------------
        # 4.2 与环境交互
        # -------------------------------------------------
        new_state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        # -------------------------------------------------
        # 4.3 存储经验到回放缓冲区
        # -------------------------------------------------
        driver.store(state, action, reward, new_state, terminated)
        
        # 更新状态
        state = new_state
        done = terminated or truncated
        
        # -------------------------------------------------
        # 4.4 定期同步目标网络 (硬更新)
        # -------------------------------------------------
        # 注意: 新的 dqn_agent.py 中已自动处理，这里保留兼容旧代码
        if timestep_n % when2sync == 0:
            driver.frozen_net.load_state_dict(driver.policy_net.state_dict())
        
        # -------------------------------------------------
        # 4.5 定期保存模型
        # -------------------------------------------------
        if timestep_n % when2save == 0:
            driver.save(driver.save_dir, 'DQN')
        
        # -------------------------------------------------
        # 4.6 定期训练网络
        # -------------------------------------------------
        # 条件: 达到学习频率 且 回放缓冲区有足够样本
        if timestep_n % when2learn == 0 and len(driver.buffer) >= batch_n:
            q_value, loss = driver.update_net(batch_n)
            loss_list.append(loss)
        
        # -------------------------------------------------
        # 4.7 定期打印报告
        # -------------------------------------------------
        if report_type == 'text':
            if timestep_n % 5000 == 0:
                print(f"\n[t={timestep_n}] Episode {episode}")
                print(f"    epsilon: {driver.epsilon:.4f}")
                print(f"    n_updates: {driver.n_updates}")
    
    # -------------------------------------------------
    # 5. 记录 episode 结果
    # -------------------------------------------------
    episode_reward_list.append(episode_reward)
    episode_length_list.append(episode_length)
    episode_loss_list.append(np.mean(loss_list) if loss_list else 0)
    
    now_time = datetime.datetime.now()
    episode_date_list.append(now_time.date().isoformat())
    episode_time_list.append(now_time.time().isoformat())
    
    # -------------------------------------------------
    # 6. 实时绘图
    # -------------------------------------------------
    if report_type == 'plot':
        plot_reward(episode, episode_reward_list, timestep_n)
    
    # -------------------------------------------------
    # 7. 定期保存日志
    # -------------------------------------------------
    if episode % when2log == 0:
        driver.write_log(
            episode_date_list,
            episode_time_list,
            episode_reward_list,
            episode_length_list,
            episode_loss_list,
            episode_epsilon_list,
            log_filename='DQN_log.csv'
        )
    
    # -------------------------------------------------
    # 8. 打印 episode 结果
    # -------------------------------------------------
    if episode % 10 == 0:
        recent_rewards = episode_reward_list[-10:]
        mean_reward = np.mean(recent_rewards)
        print(f"Episode {episode}/{play_n_episodes} | "
              f"Reward: {episode_reward:.1f} | "
              f"Mean(10): {mean_reward:.1f} | "
              f"Steps: {episode_length} | "
              f"Epsilon: {driver.epsilon:.4f}")


# ================================================================================
# 8. 训练结束，评估
# ================================================================================
print("\n" + "=" * 50)
print("训练完成！开始评估...")
print("=" * 50)

def evaluate_agent(agent, num_episodes=5, render=False):
    """
    评估训练好的智能体
    
    评估时:
    - epsilon = 0: 完全利用，不探索
    - 使用固定种子保证可重复性
    - 计算平均得分
    """
    # 创建评估环境
    render_mode = "human" if render else "rgb_array"
    eval_env = gym.make("CarRacing-v2", continuous=False, render_mode=render_mode)
    eval_env = SkipFrame(eval_env, skip=4)
    eval_env = GrayScaleObservation(eval_env)
    eval_env = ResizeObservation(eval_env, (84, 84))
    eval_env = FrameStack(eval_env, num_stack=4)
    
    # 评估时关闭探索
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


# 运行评估
avg_score = evaluate_agent(driver, num_episodes=5, render=False)

print("=" * 50)
print(f"评估完成！平均得分: {avg_score:.1f}")
if avg_score >= 900:
    print("🎉 优秀！智能体已完全掌握赛道！")
elif avg_score >= 700:
    print("👍 良好！智能体可以完成比赛")
elif avg_score >= 400:
    print("📈 一般，需要更多训练")
else:
    print("⚠️ 表现不佳，建议继续训练或调整超参数")
print("=" * 50)


# ================================================================================
# 9. 保存最终模型和日志
# ================================================================================
print("\n正在保存最终模型...")
driver.save(driver.save_dir, 'DQN_final')
driver.write_log(
    episode_date_list,
    episode_time_list,
    episode_reward_list,
    episode_length_list,
    episode_loss_list,
    episode_epsilon_list,
    log_filename='DQN_log.csv'
)

env.close()
plt.ioff()
print("\n训练脚本执行完毕！")
