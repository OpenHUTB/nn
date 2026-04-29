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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
import torch
import datetime
import csv
from pathlib import Path

import gymnasium as gym
import gymnasium.wrappers as gym_wrap
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time

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
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max-timesteps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--report", type=str, default="text", choices=["plot", "text", "none"])
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--log-filename", type=str, default="DoubleDQN_log.csv")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--render-eval", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dueling", type=int, default=1, choices=[0, 1])
    parser.add_argument("--amp", type=int, default=1, choices=[0, 1])
    parser.add_argument("--normalize-obs", type=int, default=1, choices=[0, 1])
    return parser.parse_args()


args = _parse_args()

if args.report == "plot":
    plt.ion()
else:
    plt.ioff()


# ================================================================================
# 1. 环境设置
# ================================================================================
print("=" * 50)
print("正在初始化环境...")
print("=" * 50)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# 创建环境
env = gym.make("CarRacing-v2", continuous=False)

# 预处理
env = SkipFrame(env, skip=4)

from gymnasium.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

env = GrayScaleObservation(env)
env = ResizeObservation(env, (84, 84))
env = FrameStack(env, num_stack=4)

# 重置环境
env.action_space.seed(args.seed)
state, info = env.reset(seed=args.seed)
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
    load_model=None,
    hyperparameter_overrides={
        "dueling": bool(args.dueling),
        "amp": bool(args.amp),
        "normalize_obs": bool(args.normalize_obs),
    }
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
batch_n = args.batch_size
play_n_episodes = args.episodes

episode_reward_list = []
episode_length_list = []
episode_loss_list = []
episode_epsilon_list = []
episode_date_list = []
episode_time_list = []
episode_steps_per_sec_list = []
episode_updates_per_sec_list = []

episode = 0
timestep_n = 0
when2learn = int(driver.hyperparameters.get("train_freq", 4))
when2log = args.log_every

report_type = None if args.report == "none" else args.report
max_timesteps = args.max_timesteps if args.max_timesteps and args.max_timesteps > 0 else None


# ================================================================================
# 4. 训练主循环
# ================================================================================
print("\n" + "=" * 50)
print("开始训练 DoubleDQN！")
print("=" * 50)

while episode < play_n_episodes and (max_timesteps is None or timestep_n < max_timesteps):
    episode += 1
    episode_reward = 0
    episode_length = 0
    loss_list = []
    update_count = 0
    episode_epsilon_list.append(driver.epsilon)
    
    state, info = env.reset(seed=args.seed + episode)
    ep_start_t = time.perf_counter()
    
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
        if max_timesteps is not None and timestep_n >= max_timesteps:
            break
        
        # 训练网络
        if timestep_n % when2learn == 0 and len(driver.buffer) >= batch_n:
            q_value, loss = driver.update_net(batch_n)
            loss_list.append(loss)
            update_count += 1
        
        # 打印训练进度
        if report_type == 'text' and timestep_n % 5000 == 0:
            print(f"\n[t={timestep_n}] Episode {episode}")
            print(f"    epsilon: {driver.epsilon:.4f}")
            print(f"    n_updates: {driver.n_updates}")
            if driver.scheduler:
                print(f"    learning rate: {driver.get_current_lr():.6f}")
    
    ep_end_t = time.perf_counter()
    ep_dt = max(ep_end_t - ep_start_t, 1e-9)
    # 记录结果
    episode_reward_list.append(episode_reward)
    episode_length_list.append(episode_length)
    episode_loss_list.append(np.mean(loss_list) if loss_list else 0)
    episode_steps_per_sec_list.append(episode_length / ep_dt)
    episode_updates_per_sec_list.append(update_count / ep_dt)
    
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
            log_filename=args.log_filename,
            extra_rows={
                "steps_per_sec": episode_steps_per_sec_list,
                "updates_per_sec": episode_updates_per_sec_list,
            }
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
              f"{lr_info} | "
              f"SPS: {episode_steps_per_sec_list[-1]:.1f} | "
              f"UPS: {episode_updates_per_sec_list[-1]:.1f}")


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
    eval_env = gym.make("CarRacing-v2", continuous=False, render_mode=render_mode)
    eval_env = SkipFrame(eval_env, skip=4)
    eval_env = GrayScaleObservation(eval_env)
    eval_env = ResizeObservation(eval_env, (84, 84))
    eval_env = FrameStack(eval_env, num_stack=4)
    
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


avg_score = None
if not args.skip_eval:
    avg_score = evaluate_agent(driver, num_episodes=args.eval_episodes, render=args.render_eval)

print("=" * 50)
if avg_score is not None:
    print(f"评估完成！平均得分: {avg_score:.1f}")
    if avg_score >= 900:
        print("🎉 优秀！DoubleDQN 表现优异！")
    elif avg_score >= 700:
        print("👍 良好！")
    elif avg_score >= 400:
        print("📈 一般")
    else:
        print("⚠️ 建议继续训练")
else:
    print("已跳过评估")
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
    log_filename=args.log_filename,
    extra_rows={
        "steps_per_sec": episode_steps_per_sec_list,
        "updates_per_sec": episode_updates_per_sec_list,
    }
)

env.close()
plt.ioff()
print("\n训练脚本执行完毕！")
