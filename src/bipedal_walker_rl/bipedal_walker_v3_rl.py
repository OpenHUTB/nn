"""
由 bipedal_walker_v3_rl.ipynb 转换为 .py（OpenHUTB 仓库规范：使用 .py 而非 .ipynb）。
自动转换自 Jupyter notebook，保留 markdown 单元作为顶部注释块。
"""


# ======================================================================
# # <center>**Table of Contents**</center>
# 
# **0. About Bipedal Walker**
# 
# **1. Import Libraries**
#    - 1.A Import Required Libraries
#    - 1.B Create Environment and Test
# 
# **2. Train Model for Normal Version with PPO**
#    - 2.A Preprocess Environment
#    - 2.B Train the Model
#    - 2.C Save the Model
#    - 2.D Evaluate the Model
# 
# **3. Train Model for Hardcore Version with PPO**
#    - 3.A Test the Environment
#    - 3.B Preprocess Environment
#    - 3.C Train the Hardcore Model
#    - 3.D Save the Hardcore Model
#    - 3.E Evaluate 3M Model
#    - 3.F Observe 3M Model in Human Render Mode
#    - 3.G Evaluate 5M Model
#    - 3.H Observe 5M Model in Human Render Mode
#    - 3.I Evaluate 7M Model
# 
# **4. 5M Hardcore Training Log Analysis**
#    - 4.1 Data Overview
#    - 4.2 Reward Trend Over Time
#    - 4.3 Episode Length Trend Over Time
#    - 4.4 Correlation Between Reward and Episode Length
#    - 4.5 Episode Length Moving Average
#    - 4.6 Recommendations for Improving Episode Length
# ======================================================================

# ======================================================================
# # <center>0. About Bidepal Walker</center>
# ======================================================================

# ======================================================================
# ## Bipedal Walker (Box2D)
# 
# The **Bipedal Walker** environment simulates a bipedal robot with 4 joints and 2 legs, where the goal is to walk across rugged, uneven terrain. The task requires the agent to balance and coordinate its movements effectively over a variety of surfaces.
# 
# ### Observation Space:
# - The observation space includes 24 continuous values, which provide detailed information on:
#   - Hull angle and angular velocity
#   - Horizontal and vertical speed
#   - Joint angles and speeds for both legs
#   - 10 LIDAR readings that measure the distances to the terrain below
# 
# ### Action Space:
# - The action space consists of 4 continuous values in the range \([-1, 1]\), each controlling the torque applied to the robot's joints:
#   - Hip and knee joints for both legs
#   
# ### Rewards:
# - **Positive Rewards**: For forward movement and maintaining balance.
# - **Negative Rewards**: Penalties are given for applying excessive torque to the joints and for falling.
# 
# ### Termination:
# - The episode ends if the robot falls or if the maximum number of steps (1600 in normal mode or 2000 in hardcore mode) is reached.
# 
# The environment is designed to challenge both learning algorithms and the agent's ability to handle continuous control tasks in varying terrains.
# 
# For more information, refer to the [Bipedal Walker documentation](https://gymnasium.farama.org/environments/box2d/bipedal_walker/).
# ======================================================================

# ======================================================================
# ---
# 
# # <center>1. Import Libaries</center>
# ======================================================================

# ======================================================================
# ## 1A) Import Libaries
# ======================================================================

# Import pandas for handling and analyzing data (e.g., log files)
import pandas as pd

# Import matplotlib for data visualization
import matplotlib.pyplot as plt

# Import necessary utility functions from env_utils
from env_utils import make_env, observe_model

# ======================================================================
# ## 1B) Create Env and Test
# ======================================================================

# Create the BipedalWalker environment with human-rendering mode enabled
env = gym.make("BipedalWalker-v3", render_mode="human")

# Reset the environment (start a new episode) - without using seed or options
obs = env.reset()

# Let the agent take random actions for 1000 steps
for _ in range(1000):
    # Take a random action sampled from the environment's action space
    action = env.action_space.sample()
    
    # Step the environment forward using the chosen action
    # The environment returns the new observation (obs), the reward, 
    # whether the episode is done (done), if it was truncated (truncated), and additional info (info)
    obs, reward, done, truncated, info = env.step(action)
    
    # If the episode is finished (either done or truncated), reset the environment for a new episode
    if done or truncated:
        obs = env.reset()

# Close the environment when finished to clean up resources
env.close()

# ======================================================================
# -----
# # <center>2. Train Model for Normal Version with PPO</center>
# ======================================================================

# ======================================================================
# ## 2A) Preprocces Enviorment
# ======================================================================

# ======================================================================
# ### Summary of `make_env.py`
# 
# This function is designed to create and wrap a Gym environment, specifically for the `BipedalWalker-v3` environment, with various configurable features:
# 
# 1. **Environment Creation**:  
#    - By default, the function creates the `BipedalWalker-v3` environment, but you can specify any Gym environment via the `env_name` parameter.
#    - The `hardcore` parameter allows enabling or disabling the hardcore mode (`True`/`False`). It defaults to `None`, meaning no hardcore mode unless specified.
# 
# 2. **Observation and Reward Normalization**:  
#    - The environment is wrapped with `VecNormalize` to normalize observations and rewards, providing more stable training.
# 
# 3. **Frame Stacking for Temporal Information**:  
#    - The function stacks the last `n_stack` observations (default is 4), which helps the agent to learn from temporal sequences.
# 
# 4. **Video Recording (Optional)**:  
#    - If `record_video=True`, the environment will record videos every 1000 steps and save them in the specified `video_folder`. The `render_mode` is automatically set to `rgb_array` for recording.
# 
# 5. **Monitor (Enabled by Default)**:  
#    - The `Monitor` wrapper logs performance metrics such as rewards and episode lengths during training. Logs are saved to the `logs` directory with a timestamp-based filename to avoid overwriting.
# 
# 6. **Vectorized Environment**:  
#    - The environment is wrapped with `DummyVecEnv` to enable vectorized operations, which are useful for faster training and model performance.
# ======================================================================

env = make_env()

# ======================================================================
# ## 2B) Train Model
# ======================================================================

# Create the PPO model with a Multi-Layer Perceptron (MLP) policy
model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=1000000)

# ======================================================================
# ## 2C) Save Model
# ======================================================================

model.save("models/ppo_bipedalwalker_1M")

del model

# ======================================================================
# ## 2D) Evaluate Model
# ======================================================================

model = PPO.load("models/ppo_bipedalwalker_1M")

env = gym.make("BipedalWalker-v3", render_mode="human")

# Evaluate the model (e.g., over 10 episodes)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

print(f"Average reward: {mean_reward} ± {std_reward}")

# ======================================================================
# **Average Reward**: 248.39 ± 112.10
#   - **Assessment**: This result indicates that the model is performing quite well overall. The average reward suggests that it has developed an effective policy and undergone a successful learning process. The high standard deviation (112.10) indicates that the model achieved significantly higher rewards in some trials while scoring lower in others, implying variability in its responses to different situations. This variability highlights the need for further analysis to understand how the model interacts with its environment.
# ======================================================================

# ======================================================================
# ## 2E) Observe Model in Human Render Mode
# ======================================================================

observe_model(model_path = 'models/ppo_bipedalwalker_1M')

# ======================================================================
# ------
# # <center>3. Train Model for Hardcore Version with PPO</center>
# ======================================================================

# ======================================================================
# ## 3A) Test Enviroment
# ======================================================================

env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="human")

# Reset the environment (start a new episode) - without using seed or options
obs = env.reset()

# Let the agent take random actions for 1000 steps
for _ in range(1000):
    # Take a random action sampled from the environment's action space
    action = env.action_space.sample()
    
    # Step the environment forward using the chosen action
    # The environment returns the new observation (obs), the reward, 
    # whether the episode is done (done), if it was truncated (truncated), and additional info (info)
    obs, reward, done, truncated, info = env.step(action)
    
    # If the episode is finished (either done or truncated), reset the environment for a new episode
    if done or truncated:
        obs = env.reset()

# Close the environment when finished to clean up resources
env.close()

# ======================================================================
# ## 3B) Preprocces Enviorment
# ======================================================================

env = make_env(hardcore=True)

# ======================================================================
# ## 3C) Train Model
# ======================================================================

# Create the PPO model with a Multi-Layer Perceptron (MLP) policy
model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=5000000)

# ======================================================================
# ## 3D) Save Model
# ======================================================================

model.save("models/ppo_bipedalwalker_hardcore_3M")

del model

# ======================================================================
# ## 3E) Evaluate Model 3M Model
# ======================================================================

model = PPO.load("models/ppo_bipedalwalker_hardcore_3M")

env = gym.make("BipedalWalker-v3", hardcore=True)

# Evaluate the model (e.g., over 10 episodes)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

print(f"Average reward: {mean_reward} ± {std_reward}")

# ======================================================================
# **Average Reward**: -28.23 ± 24.82
#   - **Assessment**: This result shows that the model is underperforming in the more challenging environment. A negative average reward indicates that the model mostly receives unfavorable feedback and struggles to achieve the target. The lower standard deviation (24.82) suggests less variability in performance, indicating that the model consistently performs poorly under difficult conditions. This may imply that the model requires more training and potentially different hyperparameter settings.
# ======================================================================

# ======================================================================
# ## 3F) Observe 3M Model in Human Render Mode
# ======================================================================

observe_model(model_path = 'models/ppo_bipedalwalker_hardcore_3M', hardcore = True)

# ======================================================================
# ## 3G) Evaluate Model 5M Model
# ======================================================================

del model

model = PPO.load("models/ppo_bipedalwalker_hardcore_5M")

env = make_env("BipedalWalker-v3", hardcore=True)

# Evaluate the model (e.g., over 100 episodes)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

print(f"Average reward: {mean_reward} ± {std_reward}")

# ======================================================================
# **Average Reward**: -10.66 ± 3.91  
# - **Assessment**: This result indicates that the model is not performing well in the environment, as evidenced by the negative average reward. A negative score suggests that the agent primarily receives penalties, reflecting its struggle to reach the desired outcomes. The standard deviation of 3.91 indicates relatively low variability in performance, meaning the model consistently underperforms rather than showing sporadic successes. This suggests that the model may benefit from further training and adjustments in hyperparameters to improve its learning effectiveness.
# ======================================================================

# ======================================================================
# ## 3H) Observe 5M Model in Human Render Mode
# ======================================================================

observe_model(model_path = 'models/ppo_bipedalwalker_hardcore_7M', hardcore = True)

# ======================================================================
# ## 3I) Evaluate Model 7M Model
# ======================================================================

model = PPO.load("models/ppo_bipedalwalker_hardcore_7M")
env = make_env("BipedalWalker-v3", hardcore=True)

# Evaluate the model (e.g., over 100 episodes)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

print(f"Average reward: {mean_reward} ± {std_reward}")

# ======================================================================
# -----
# # <center>4. 5m Hardcore Training Log Analysis </center>
# ======================================================================

# ======================================================================
# This section provides an in-depth analysis of the 5m hardcore training logs. The analysis focuses on key metrics such as reward, episode length, and their correlation, with visualizations to help interpret the results effectively.
# ======================================================================

# ======================================================================
# ## 4A) Data Overview
# ======================================================================

# ======================================================================
# The training log contains three key columns:
# - `reward`: The reward obtained by the agent in each episode.
# - `episode_length`: The length (number of steps) of each episode.
# - `time`: The time elapsed during the training process.
# 
# We start by loading the data and cleaning it for further analysis.
# ======================================================================

# Load the dataset
data = pd.read_csv('logs/5m_hardcore.monitor.csv', skiprows=1)
data.columns = ['reward', 'episode_length', 'time']
data_clean = data.dropna()

# Display the first few rows
data_clean.head()

# ======================================================================
# ## 4B) Reward Trend Over Time
# ======================================================================

# ======================================================================
# In the first step, we visualize how the reward evolves over time during training. This helps in understanding how well the agent is performing over the course of training.
# ======================================================================

# Plot reward over time
plt.figure(figsize=(10, 5))
plt.plot(data_clean['time'], data_clean['reward'], label='Reward')
plt.title('Reward Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Reward')
plt.grid(True)
plt.show()

# ======================================================================
# **Insight:**
# The reward fluctuates significantly over time but shows a general stabilization trend. This suggests that the agent may have reached a steady learning phase where its performance remains stable with minor variations.
# ======================================================================

# ======================================================================
# ## 4C) Episode Length Trend Over Time
# ======================================================================

# ======================================================================
# Next, we examine how the episode length changes over time. This metric helps understand how long the agent survives or performs in each episode.
# ======================================================================

# Plot episode length over time
plt.figure(figsize=(10, 5))
plt.plot(data_clean['time'], data_clean['episode_length'], label='Episode Length', color='orange')
plt.title('Episode Length Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Episode Length')
plt.grid(True)
plt.show()

# ======================================================================
# **Insight:**
# The episode length tends to remain relatively high throughout the training, with occasional dips. This indicates that the agent consistently completes longer episodes, which could mean it is learning to survive longer in the environment.
# ======================================================================

# ======================================================================
# ## 4D) Correlation Between Reward and Episode Length
# ======================================================================

# ======================================================================
# A key question is whether there is a correlation between the reward and the episode length. To investigate this, we calculate the correlation coefficient between these two variables.
# ======================================================================

# Calculate the correlation between reward and episode length
correlation = data_clean['reward'].corr(data_clean['episode_length'])
print(f'Correlation between reward and episode length: {correlation:.2f}')

# ======================================================================
# **Insight:**
# The calculated correlation is 0.89, which indicates a strong positive correlation. This means that as the episode length increases, the reward also tends to increase. Essentially, the longer the agent survives, the more reward it earns.
# ======================================================================

# ======================================================================
# ## 4E) Episode Length Moving Average
# ======================================================================

# ======================================================================
# To smooth out the episode length data and observe longer-term trends, we use a moving average with a window size of 50.
# ======================================================================

# Moving average of episode length
window_size = 50
data_clean['episode_length_ma'] = data_clean['episode_length'].rolling(window=window_size).mean()

# Plot episode length with moving average
plt.figure(figsize=(10, 5))
plt.plot(data_clean['time'], data_clean['episode_length'], label='Episode Length', color='orange')
plt.plot(data_clean['time'], data_clean['episode_length_ma'], label=f'Moving Average ({window_size} windows)', color='blue')
plt.title('Episode Length Over Time with Moving Average')
plt.xlabel('Time (seconds)')
plt.ylabel('Episode Length')
plt.legend()
plt.grid(True)
plt.show()

# ======================================================================
# **Insight:**
# The moving average reveals that the episode length has a slight upward trend over time, indicating that the agent may be gradually learning to perform longer episodes as training progresses.
# ======================================================================

# ======================================================================
# ## 4F) Recommendations for Improving Episode Length
# ======================================================================

# ======================================================================
# Based on the analysis, here are some strategies to potentially increase the episode length and improve agent performance:
# 
# **1. Adjust Learning Rate:** Consider lowering the learning rate to allow for more gradual improvements.
# 
# **2. Modify Reward Function:** Adjust the reward structure to incentivize the agent for surviving longer in each episode.
# 
# **3. Increase Exploration:** Encourage more exploration by adjusting the epsilon in ε-greedy policies or employing curiosity-driven methods.
# 
# **4. Extend Training Duration:** Increasing the number of timesteps during training may allow the agent to learn better strategies for longer survival.
# 
# **5. Use Experience Replay:** Implementing experience replay could help the agent learn from past episodes and improve over time.
# 
# By following these recommendations, the agent’s performance could be enhanced, leading to longer episode durations and improved rewards.
# ======================================================================

# ======================================================================
# ---
# 
# This write-up includes Markdown text for Jupyter, along with code snippets for generating visualizations and insights. It summarizes key findings such as reward trends, episode length behavior, and actionable steps to improve training.
# ======================================================================
