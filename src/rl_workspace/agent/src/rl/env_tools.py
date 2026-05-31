"""
环境包装器和预处理工具

提供各种环境包装器和状态/动作预处理功能
"""

import numpy as np
from typing import Tuple, Optional, List, Callable
import gymnasium as gym


class NormalizedEnv:
    """环境奖励归一化包装器"""
    
    def __init__(self, env, clip_rewards: float = 10.0, normalize_rewards: bool = True):
        self.env = env
        self.clip_rewards = clip_rewards
        self.normalize_rewards = normalize_rewards
        self.returns = None
        self.count = 0
    
    def reset(self):
        if self.normalize_rewards and self.returns is None:
            self.returns = []
        return self.env.reset()
    
    def step(self, action):
        next_state, reward, done, trunc, info = self.env.step(action)
        
        if self.normalize_rewards:
            self.returns.append(reward)
            reward = np.clip(reward, -self.clip_rewards, self.clip_rewards)
        
        return next_state, reward, done, trunc, info
    
    def __getattr__(self, name):
        return getattr(self.env, name)


class FrameStack:
    """帧堆叠包装器（用于Atari等图像环境）"""
    
    def __init__(self, env, num_stack: int = 4):
        self.env = env
        self.num_stack = num_stack
        self.frames = None
    
    def reset(self):
        state = self.env.reset()[0]
        self.frames = [state] * self.num_stack
        return np.concatenate(self.frames, axis=-1)
    
    def step(self, action):
        state, reward, done, trunc, info = self.env.step(action)
        self.frames.append(state)
        self.frames = self.frames[-self.num_stack:]
        return np.concatenate(self.frames, axis=-1), reward, done, trunc, info
    
    def __getattr__(self, name):
        return getattr(self.env, name)


class ActionRepeat:
    """动作重复包装器"""
    
    def __init__(self, env, repeat: int = 4):
        self.env = env
        self.repeat = repeat
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        total_reward = 0.0
        for _ in range(self.repeat):
            state, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done or trunc:
                break
        return state, total_reward, done, trunc, info
    
    def __getattr__(self, name):
        return getattr(self.env, name)


class RewardScaling:
    """奖励缩放"""
    
    def __init__(self, env, scale_factor: float = 1.0):
        self.env = env
        self.scale_factor = scale_factor
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        state, reward, done, trunc, info = self.env.step(action)
        return state, reward * self.scale_factor, done, trunc, info
    
    def __getattr__(self, name):
        return getattr(self.env, name)


class StatePreprocessor:
    """状态预处理器"""
    
    def __init__(self, state_dim: int, normalize: bool = False):
        self.state_dim = state_dim
        self.normalize = normalize
        self.running_mean = None
        self.running_var = None
        self.count = 0
    
    def update(self, states: np.ndarray):
        """更新统计量"""
        if self.running_mean is None:
            self.running_mean = np.zeros_like(states[0])
            self.running_var = np.ones_like(states[0])
        
        batch_mean = np.mean(states, axis=0)
        batch_var = np.var(states, axis=0)
        batch_count = len(states)
        
        delta = batch_mean - self.running_mean
        total_count = self.count + batch_count
        
        self.running_mean += delta * batch_count / total_count
        self.running_var = (
            self.running_var * self.count + batch_var * batch_count + 
            delta ** 2 * self.count * batch_count / total_count
        ) / total_count
        
        self.count = total_count
    
    def transform(self, state: np.ndarray) -> np.ndarray:
        """应用预处理"""
        if self.normalize and self.running_mean is not None:
            state = (state - self.running_mean) / (np.sqrt(self.running_var) + 1e-8)
        return state


class ActionScaler:
    """动作空间缩放器"""
    
    def __init__(self, action_space):
        self.action_space = action_space
        self.low = action_space.low if hasattr(action_space, 'low') else None
        self.high = action_space.high if hasattr(action_space, 'high') else None
    
    def scale(self, action: np.ndarray) -> np.ndarray:
        """缩放到环境动作空间"""
        if self.low is not None and self.high is not None:
            return self.low + (action + 1) * 0.5 * (self.high - self.low)
        return action
    
    def unscale(self, action: np.ndarray) -> np.ndarray:
        """从环境动作空间反缩放"""
        if self.low is not None and self.high is not None:
            return 2 * (action - self.low) / (self.high - self.low) - 1
        return action


class EpisodeStatistics:
    """回合统计"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_reward = 0
        self.current_length = 0
    
    def update(self, reward: float):
        self.current_reward += reward
        self.current_length += 1
    
    def end_episode(self):
        self.episode_rewards.append(self.current_reward)
        self.episode_lengths.append(self.current_length)
        self.current_reward = 0
        self.current_length = 0
    
    def get_mean_reward(self, window: int = 100) -> float:
        if not self.episode_rewards:
            return 0.0
        return np.mean(self.episode_rewards[-window:])
    
    def get_mean_length(self, window: int = 100) -> float:
        if not self.episode_lengths:
            return 0.0
        return np.mean(self.episode_lengths[-window:])


class WandbLogger:
    """Weights & Biases 日志记录器（可选）"""
    
    def __init__(self, project_name: str, config: dict = None, enabled: bool = False):
        self.enabled = enabled
        self.project_name = project_name
        self.config = config
        self.step = 0
        
        if self.enabled:
            try:
                import wandb
                wandb.init(project=project_name, config=config)
                self.wandb = wandb
            except ImportError:
                print("wandb 未安装，将禁用日志记录")
                self.enabled = False
    
    def log(self, metrics: dict, step: Optional[int] = None):
        if not self.enabled:
            return
        
        if step is not None:
            self.step = step
        
        self.wandb.log(metrics, step=self.step)
    
    def finish(self):
        if self.enabled:
            self.wandb.finish()


class TensorboardLogger:
    """TensorBoard 日志记录器"""
    
    def __init__(self, log_dir: str = "runs", enabled: bool = True):
        self.enabled = enabled
        self.log_dir = log_dir
        self.step = 0
        
        if self.enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir)
            except ImportError:
                try:
                    from tensorboardX import SummaryWriter
                    self.writer = SummaryWriter(log_dir)
                except ImportError:
                    print("tensorboard 未安装，将禁用日志记录")
                    self.enabled = False
    
    def log(self, metrics: dict, step: Optional[int] = None):
        if not self.enabled:
            return
        
        if step is not None:
            self.step = step
        
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, self.step)
        
        self.step += 1
    
    def close(self):
        if self.enabled:
            self.writer.close()


def make_env(env_name: str, **kwargs) -> gym.Env:
    """创建环境工厂"""
    env = gym.make(env_name, **kwargs)
    return env


def evaluate_policy(
    env: gym.Env,
    agent,
    num_episodes: int = 10,
    render: bool = False,
    deterministic: bool = True
) -> Tuple[float, float]:
    """
    评估策略
    
    返回:
        mean_reward: 平均奖励
        std_reward: 奖励标准差
    """
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        episode_reward = 0
        
        while not done:
            if render:
                env.render()
            
            if hasattr(agent, 'select_action'):
                action = agent.select_action(state, training=not deterministic)
            else:
                action = agent.get_action(state, training=not deterministic)
            
            if isinstance(action, tuple):
                action = action[0]
            
            state, reward, done, trunc, info = env.step(action)
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
    
    return np.mean(episode_rewards), np.std(episode_rewards)
