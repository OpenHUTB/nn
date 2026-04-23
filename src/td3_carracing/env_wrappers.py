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
            low=0, high=1, shape=(h, w, stack), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs for _ in range(self.stack)]
        return self._get_state(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.pop(0)
        self.frames.append(obs)
        return self._get_state(), reward, terminated, truncated, info

    def _get_state(self):
        state = np.concatenate(self.frames, axis=-1)
        state = state.transpose(2, 0, 1)
        return state

class SmoothActionWrapper(gym.Wrapper):
    def __init__(self, env, alpha=0.9):
        super().__init__(env)
        self.alpha = alpha
        self.last_action = None

    def step(self, action):
        if self.last_action is not None:
            action = self.alpha * action + (1 - self.alpha) * self.last_action
            action[0] = np.clip(action[0], self.last_action[0] - 0.08, self.last_action[0] + 0.08)
        self.last_action = action.copy()
        return self.env.step(action)

    def reset(self, **kwargs):
        self.last_action = None
        return self.env.reset(**kwargs)

class TrackBoundaryWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.off_track_penalty = 2.0       # 出赛道惩罚大幅提高
        self.small_steer_penalty = 0.05    # 小角度乱打方向惩罚
        self.max_steer = 0.55               # 限制转向更温和
        self.on_track_reward = 0.4          # 在赛道上奖励
        self.last_progress = 0.0

    def step(self, action):
        # 强制限制转向，防止冲出去
        action[0] = np.clip(action[0], -self.max_steer, self.max_steer)

        obs, reward, terminated, truncated, info = self.env.step(action)
        speed = info.get('speed', 0.0)

        # ===== 出赛道检测 =====
        if reward < -0.5:  # CarRacing 内部出赛道会给负奖励
            reward -= self.off_track_penalty  # 额外重罚
            # 出赛道立即减速，强制纠正
            action[1] = 0.0
            action[2] = 0.2

        # ===== 在赛道内奖励 =====
        if reward > -0.1 and speed > 0.5:
            reward += self.on_track_reward

        # ===== 禁止原地小角度乱摆 =====
        if abs(action[0]) < 0.08 and speed < 1.0:
            reward -= self.small_steer_penalty

        # ===== 出赛道直接终止 =====
        if reward < -1.0:
            truncated = True
            reward -= 3.0

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.last_progress = 0.0
        return self.env.reset(**kwargs)

def wrap_env(env):
    env = SkipFrame(env, skip=4)
    env = PreProcessObs(env)
    env = StackFrames(env, stack=4)
    env = TrackBoundaryWrapper(env)
    env = SmoothActionWrapper(env, alpha=0.9)
    return env