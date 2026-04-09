"""观测处理的环境包装器"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from typing import Tuple, Dict, Any


class FrameStackWrapper(gym.Wrapper):
    """堆叠多个连续帧以捕获时间信息"""

    def __init__(self, env: gym.Env, num_frames: int = 4):
        """
        初始化帧堆叠包装器。

        Args:
            env: 基础环境
            num_frames: 要堆叠的帧数
        """
        super().__init__(env)
        self.num_frames = num_frames

        # Create stacked observation space
        assert isinstance(env.observation_space, spaces.Box)

        old_shape = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=(*old_shape[:-1], old_shape[-1] * num_frames),
            dtype=env.observation_space.dtype,
        )

        # Frame buffer
        self.frames = deque(maxlen=num_frames)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """重置环境和帧缓冲区"""
        obs, info = self.env.reset(seed=seed, options=options)

        # Clear frame buffer and fill with initial observation
        for _ in range(self.num_frames):
            self.frames.append(obs)

        return self._stack_frames(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行动作并堆叠帧"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._stack_frames(), reward, terminated, truncated, info

    def _stack_frames(self) -> np.ndarray:
        """沿最后一维堆叠帧"""
        return np.concatenate(list(self.frames), axis=-1)


class NormalizeObsWrapper(gym.ObservationWrapper):
    """将观测值归一化为 [0, 1] 范围"""

    def __init__(self, env: gym.Env):
        """
        初始化观测值归一化包装器。

        Args:
            env: 基础环境
        """
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """归一化观测值"""
        # 从 [0, 255] 归一化为 [0, 1]
        if obs.dtype == np.uint8:
            return obs.astype(np.float32) / 255.0
        return obs


class ResizeWrapper(gym.ObservationWrapper):
    """将观测值调整为目标形状"""

    def __init__(self, env: gym.Env, target_shape: Tuple[int, int] = (84, 84)):
        """
        初始化调整大小包装器。

        Args:
            env: 基础环境
            target_shape: 目标 (H, W) 形状
        """
        super().__init__(env)

        assert isinstance(env.observation_space, spaces.Box)

        self.target_shape = target_shape

        # Update observation space
        old_obs_shape = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=(*target_shape, old_obs_shape[-1]),
            dtype=env.observation_space.dtype,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """调整观测值大小"""
        from PIL import Image

        # Extract spatial dimensions
        H, W = obs.shape[:2]
        C = obs.shape[-1] if len(obs.shape) == 3 else 1

        # Resize each channel
        if C == 1:
            # Single channel
            img = Image.fromarray(obs.squeeze(), mode="L")
            img_resized = img.resize(self.target_shape[::-1], Image.Resampling.LANCZOS)
            return np.array(img_resized, dtype=obs.dtype).reshape(*self.target_shape, 1)
        else:
            # Multi-channel
            img = Image.fromarray(obs, mode="RGB" if C == 3 else f"{C}")
            img_resized = img.resize(self.target_shape[::-1], Image.Resampling.LANCZOS)
            return np.array(img_resized, dtype=obs.dtype)


class ClipRewardWrapper(gym.RewardWrapper):
    """将奖励剪裁到 [-1, 1] 范围"""

    def __init__(self, env: gym.Env):
        """初始化奖励剪裁包装器"""
        super().__init__(env)

    def reward(self, reward: float) -> float:
        """剪裁奖励"""
        return np.clip(reward, -1, 1)


class ActionRepeatWrapper(gym.Wrapper):
    """为多个步骤重复动作"""

    def __init__(self, env: gym.Env, repeat: int = 4):
        """
        初始化动作重复包装器。

        Args:
            env: 基础环境
            repeat: 重复该动作的次数
        """
        super().__init__(env)
        self.repeat = repeat

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """重复动作并累积奖励"""
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for i in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


def create_wrapped_env(
    env: gym.Env,
    frame_stack: int = 4,
    normalize_obs: bool = True,
    clip_reward: bool = False,
    action_repeat: int = 1,
) -> gym.Env:
    """
    创建具有标准预处理的包装环境。

    Args:
        env: 基础环境
        frame_stack: 要堆叠的帧数（0 = 不堆叠）
        normalize_obs: 是否将观测值归一化为 [0, 1]
        clip_reward: 是否将奖励剪裁到 [-1, 1]
        action_repeat: 每个动作重复的次数

    返回值:
        包装环境
    """
    # 按顺序应用包装器
    if frame_stack > 1:
        env = FrameStackWrapper(env, num_frames=frame_stack)

    if normalize_obs:
        env = NormalizeObsWrapper(env)

    if clip_reward:
        env = ClipRewardWrapper(env)

    if action_repeat > 1:
        env = ActionRepeatWrapper(env, repeat=action_repeat)

    return env
