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

from double_dqn import plot_reward
from double_dqn import DoubleDQNAgent

# Adjust path to import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from double_dqn_model.double_dqn import DoubleDQNAgent
from dqn_model.DQN_model import SkipFrame

from double_dqn_model.double_dqn import plot_reward

from gymnasium.spaces import Box
from tensordict import TensorDict
from torch import nn
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# Environment setup
env = gym.make("CarRacing-v3", continuous=False)
env = SkipFrame(env, skip=4)
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
env = GrayscaleObservation(env)
env = ResizeObservation(env, (84, 84))
env = FrameStackObservation(env, stack_size=4)

# Initialize environment and get state shape
state, info = env.reset()
action_n = env.action_space.n
print(f"Action space size: {action_n}")
print(f"Action space details: {env.action_space}")

# Ensure correct config path
config_path = Path(__file__).parent.parent / 'configs' / 'double_dqn.yaml'

# Agent initialization
driver = DoubleDQNAgent(
    state_space_shape=state.shape,
    action_n=action_n,
    config_path=config_path,
    load_state=False,  # Explicitly set to False if not loading a model
    load_model=None    # Set to model filename if loading
)

# Verify config loaded
print(f"Using tau: {driver.tau}")
print(f"Target update every: {driver.update_target_every} steps")

# Training parameters
batch_n = 32
play_n_episodes = 2000  # Keep low for quick testing, adjust as needed
episode_epsilon_list = []
episode_reward_list = []
episode_length_list = []
episode_loss_list = []
episode_date_list = []
episode_time_list = []
episode = 0
timestep_n = 2
when2learn = 4  # in timesteps
when2sync = 5000  # in timesteps
when2save = 100000  # in timesteps
when2report = 5000  # in timesteps
when2eval = 50000  # in timesteps
when2log = 10  # in episodes
report_type = 'plot'  # 'text', 'plot', None

# Training loop
while episode <= play_n_episodes:
    episode += 1
    episode_reward = 0
    episode_length = 0
    updating = True
    loss_list = []
    episode_epsilon_list.append(driver.epsilon)

    while updating:
        timestep_n += 1
        episode_length += 1

        action = driver.take_action(state)
        new_state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        driver.store(state, action, reward, new_state, terminated)
        state = new_state
        updating = not (terminated or truncated)

        if timestep_n % when2sync == 0:
            # Sync target network manually (Double DQN handles this internally, but we can enforce it)
            driver.target_net.load_state_dict(driver.policy_net.state_dict())

        if timestep_n % when2save == 0:
            driver.save(driver.save_dir, 'DoubleDQN')

        if timestep_n % when2learn == 0 and len(driver.buffer) >= batch_n:
            q, loss = driver.update_net(batch_n)
            loss_list.append(loss)

        if timestep_n % when2report == 0 and report_type == 'text':
            print(f'Report: {timestep_n} timestep')
            print(f'    episodes: {episode}')
            print(f'    n_updates: {driver.n_updates}')
            print(f'    epsilon: {driver.epsilon}')

        if timestep_n % when2eval == 0 and report_type == 'text':
            rewards_tensor = torch.tensor(episode_reward_list, dtype=torch.float)
            eval_reward = torch.clone(rewards_tensor[-50:])
            mean_eval_reward = round(torch.mean(eval_reward).item(), 2)
            std_eval_reward = round(torch.std(eval_reward).item(), 2)

            lengths_tensor = torch.tensor(episode_length_list, dtype=torch.float)
            eval_length = torch.clone(lengths_tensor[-50:])
            mean_eval_length = round(torch.mean(eval_length).item(), 2)
            std_eval_length = round(torch.std(eval_length).item(), 2)

            print(f'Evaluation: {timestep_n} timestep')
            print(f'    reward {mean_eval_reward}±{std_eval_reward}')
            print(f'    episode length {mean_eval_length}±{std_eval_length}')
            print(f'    episodes: {episode}')
            print(f'    n_updates: {driver.n_updates}')
            print(f'    epsilon: {driver.epsilon}')

    state, info = env.reset()

    episode_reward_list.append(episode_reward)
    episode_length_list.append(episode_length)
    episode_loss_list.append(np.mean(loss_list) if loss_list else 0)
    now_time = datetime.datetime.now()
    episode_date_list.append(now_time.date().strftime('%Y-%m-%d'))
    episode_time_list.append(now_time.time().strftime('%H:%M:%S'))

    if report_type == 'plot':
        draw_check = plot_reward(episode, episode_reward_list, timestep_n)

    if episode % when2log == 0:
        driver.write_log(
            episode_date_list,
            episode_time_list,
            episode_reward_list,
            episode_length_list,
            episode_loss_list,
            episode_epsilon_list,
            log_filename='DoubleDQN_log_test.csv'
        )

if report_type == 'text':
    rewards_tensor = torch.tensor(episode_reward_list, dtype=torch.float)
    eval_reward = torch.clone(rewards_tensor[-100:])
    mean_eval_reward = round(torch.mean(eval_reward).item(), 2)
    std_eval_reward = round(torch.std(eval_reward).item(), 2)

    lengths_tensor = torch.tensor(episode_length_list, dtype=torch.float)
    eval_length = torch.clone(lengths_tensor[-100:])
    mean_eval_length = round(torch.mean(eval_length).item(), 2)
    std_eval_length = round(torch.std(eval_length).item(), 2)

    print(f'Final evaluation: {timestep_n} timestep')
    print(f'    reward {mean_eval_reward}±{std_eval_reward}')
    print(f'    episode length {mean_eval_length}±{std_eval_length}')
    print(f'    episodes: {episode}')
    print(f'    n_updates: {driver.n_updates}')
    print(f'    epsilon: {driver.epsilon}')

driver.save(driver.save_dir, 'DoubleDQN')
driver.write_log(
    episode_date_list,
    episode_time_list,
    episode_reward_list,
    episode_length_list,
    episode_loss_list,
    episode_epsilon_list,
    log_filename='DoubleDQN_log_test.csv'
)
env.close()
plt.ioff()

# Evaluation Mode
def evaluate_agent(agent, num_episodes=2, render=True):
    """Evaluate the trained agent with visualization"""
    if render:
        env = gym.make("CarRacing-v3", continuous=False, render_mode="human")
    else:
        env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
    
    env = SkipFrame(env, skip=4)  # Use SkipFrame from DQN_model.py
    env = GrayscaleObservation(env)
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, stack_size=4)
    
    agent.epsilon = 0  # Disable exploration
    seeds_list = [i for i in range(num_episodes)]
    scores = []
    
    for episode, seed in enumerate(seeds_list):
        state, info = env.reset(seed=seed)
        score = 0
        updating = True
        
        while updating:
            action = agent.take_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            score += reward
            updating = not (terminated or truncated)
        
        scores.append(score)
        print(f"Evaluation Episode {episode+1}/{num_episodes} | Seed: {seed} | Score: {score:.1f}")
    
    env.close()
    return np.mean(scores)

# Run evaluation after training
print("\n=== Starting Evaluation ===")
avg_score = evaluate_agent(driver, num_episodes=2)
print(f"\nAverage evaluation score: {avg_score:.1f}")
plt.show()