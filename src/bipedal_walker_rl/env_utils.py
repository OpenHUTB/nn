import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

def make_env(env_name="BipedalWalker-v3", hardcore=None, n_stack=4, clip_obs=10.0, render_mode=None, record_video=False, video_folder='videos', use_monitor=False, logs_dir='logs'):
    """
    Create and wrap the environment for BipedalWalker with optional hardcore mode,
    vectorized operations, normalization, frame stacking, rendering options, video recording, and monitoring.
    
    Args:
    - env_name (str): Name of the environment to create. Default is 'BipedalWalker-v3'.
    - hardcore (bool or None): Whether to enable hardcore mode in the environment. Default is None.
    - n_stack (int): Number of frames to stack. Default is 4.
    - clip_obs (float): Value to clip observations to avoid outliers. Default is 10.0.
    - render_mode (str or None): Render mode for the environment. Can be 'human', 'rgb_array', etc. Default is None.
    - record_video (bool): Whether to record video during the environment execution. Default is False.
    - video_folder (str): Directory where video recordings will be saved. Default is 'videos'.
    - use_monitor (bool): Whether to wrap the environment with Monitor for logging. Default is False.
    - logs_dir (str): Directory where monitor logs will be saved. Default is 'logs'.
    
    Returns:
    - env: A wrapped and prepared environment.
    """
    
    # If recording video, enforce render_mode to 'rgb_array'
    if record_video:
        render_mode = 'rgb_array'
        print(f"Video recording is enabled. Setting render_mode to 'rgb_array'.")
    else:
        print(f"Video recording is not enabled. Using render_mode: {render_mode or 'None'}")
    
    # Create the environment with the specified render mode and hardcore mode (if provided)
    if hardcore is not None:
        env = gym.make(env_name, hardcore=hardcore, render_mode=render_mode)
        print(f"Creating environment with hardcore={hardcore}")
    else:
        env = gym.make(env_name, render_mode=render_mode)
        print(f"Creating environment without hardcore mode.")
    
    # If monitor logging is enabled, wrap the environment with Monitor
    if use_monitor:
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{timestamp}"
        monitor_log_path = os.path.join(logs_dir, log_filename)
        
        env = Monitor(env, filename=monitor_log_path)
        print(f"Monitoring enabled. Logs will be saved to: {monitor_log_path}")
    else:
        print("Monitor is disabled.")
    
    # Wrap the environment in a DummyVecEnv to enable vectorized operations
    env = DummyVecEnv([lambda: env])
    
    # Normalize observations and rewards in the environment
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=clip_obs)
    
    # Stack the last n_stack observations
    env = VecFrameStack(env, n_stack=n_stack)
    
    # If recording video, add the video recording wrapper
    if record_video:
        os.makedirs(video_folder, exist_ok=True)
        env = VecVideoRecorder(env, video_folder, record_video_trigger=lambda x: x % 1000 == 0, video_length=200)
        print(f"Video recordings will be saved in the folder: {video_folder}")
    
    return env
    print("Environment created.")


    


def observe_model(model_path, n_eval_episodes=5, hardcore=False):
    """
    Load the trained PPO model and evaluate it in the BipedalWalker environment.
    It also checks for VecFrameStack and VecNormalize wrappers and adds them if needed.

    Args:
    - model_path (str): Path to the trained model.
    - n_eval_episodes (int): Number of episodes to evaluate. Default is 5.
    - hardcore (bool): If True, use the hardcore version of the environment. Default is False.

    Returns:
    - mean_reward: The mean reward obtained during evaluation.
    """
    
    # Load the trained PPO model
    model = PPO.load(model_path)

    # Check if hardcore mode is required, and create the appropriate environment
    env_id = "BipedalWalkerHardcore-v3" if hardcore else "BipedalWalker-v3"
    env = gym.make(env_id, render_mode='human')

    # Wrap the environment in DummyVecEnv for vectorized operations
    env = DummyVecEnv([lambda: env])

    # If VecNormalize is used, wrap the environment in VecNormalize
    if isinstance(model.get_env(), VecNormalize):
        env = VecNormalize(env)
        print("VecNormalize used.")

    # Check model's observation space to add VecFrameStack if necessary
    observation_space_shape = model.observation_space.shape[0] if model.observation_space else None
    if observation_space_shape == 96:
        # If the model was trained with 4-frame stacking, add VecFrameStack
        env = VecFrameStack(env, n_stack=4)
        print("VecFrameStack(n_stack=4) used.")

    # Evaluate the model and return the mean and standard deviation of the reward
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)

    # Close the environment after evaluation
    env.close()

    print(f"Mean Reward: {mean_reward} +/- {std_reward}")
    return mean_reward, std_reward