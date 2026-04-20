import gymnasium as gym
import cv2
import numpy as np


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class PreProcessObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(42, 42, 1), dtype=np.float32
        )

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (42, 42), interpolation=cv2.INTER_AREA)
        obs = obs / 255.0
        obs = obs[..., None]
        return obs


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, stack=4):
        super().__init__(env)
        self.stack = stack
        self.frames = []
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(stack, h, w), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs for _ in range(self.stack)]
        return self._get_state(), info

    def observation(self, obs):
        self.frames.pop(0)
        self.frames.append(obs)
        return self._get_state()

    def _get_state(self):
        state = np.concatenate(self.frames, axis=-1)
        state = state.transpose(2, 0, 1)
        return state


class SmoothActionWrapper(gym.Wrapper):
    """动作平滑包装器 - 让车辆运行更平滑"""

    def __init__(self, env, alpha=0.7):
        super().__init__(env)
        self.alpha = alpha  # 平滑系数，越大越接近原始动作
        self.last_action = None

    def step(self, action):
        if self.last_action is not None:
            # 指数移动平均平滑
            action = self.alpha * action + (1 - self.alpha) * self.last_action
        self.last_action = action.copy()
        return self.env.step(action)

    def reset(self, **kwargs):
        self.last_action = None
        return self.env.reset(**kwargs)


def wrap_env(env):
    env = SkipFrame(env)
    env = PreProcessObs(env)
    env = StackFrames(env)
    env = SmoothActionWrapper(env, alpha=0.7)  # 添加动作平滑
    return env