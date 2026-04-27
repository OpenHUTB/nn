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
            low=0, high=1, shape=(84, 84, 1), dtype=np.float32
        )

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
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
    def __init__(self, env, alpha=0.85, max_steer_change=0.15):
        super().__init__(env)
        self.alpha = alpha
        self.max_steer_change = max_steer_change
        self.last_action = None

    def step(self, action):
        if self.last_action is not None:
            action = self.alpha * action + (1 - self.alpha) * self.last_action
            action[0] = np.clip(action[0],
                                self.last_action[0] - self.max_steer_change,
                                self.last_action[0] + self.max_steer_change)
        self.last_action = action.copy()
        return self.env.step(action)

    def reset(self, **kwargs):
        self.last_action = None
        return self.env.reset(**kwargs)

class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.max_steer = 0.6
        self.consecutive_off_track = 0
        self.max_off_track_steps = 5

    def step(self, action):
        action[0] = np.clip(action[0], -self.max_steer, self.max_steer)

        obs, reward, terminated, truncated, info = self.env.step(action)
        speed = info.get('speed', 0.0)

        # 计算原始奖励
        original_reward = reward
        shaped_reward = 0.0

        # ===== 速度奖励 =====
        if speed > 0.5:
            speed_reward = min(speed / 10.0, 0.3)
            shaped_reward += speed_reward

        # ===== 出赛道检测与惩罚 =====
        if original_reward < -0.5:
            self.consecutive_off_track += 1
            off_track_penalty = 1.0 + self.consecutive_off_track * 0.5
            shaped_reward -= off_track_penalty

            if self.consecutive_off_track >= self.max_off_track_steps:
                truncated = True
                shaped_reward -= 5.0
        else:
            self.consecutive_off_track = 0
            if original_reward > -0.1:
                shaped_reward += 0.15

        # ===== 平滑驾驶奖励 =====
        steer_magnitude = abs(action[0])
        if steer_magnitude < 0.05 and speed > 1.0:
            shaped_reward += 0.05
        elif steer_magnitude > 0.3 and speed > 2.0:
            shaped_reward -= 0.1

        # ===== 高速平稳驾驶额外奖励 =====
        if speed > 2.0 and steer_magnitude < 0.2 and original_reward > -0.1:
            shaped_reward += 0.2

        # ===== 低速惩罚（防止龟速行驶） =====
        if speed < 0.3 and original_reward > -0.1:
            shaped_reward -= 0.05

        total_reward = original_reward + shaped_reward

        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.consecutive_off_track = 0
        return self.env.reset(**kwargs)

def wrap_env(env):
    env = SkipFrame(env, skip=4)
    env = PreProcessObs(env)
    env = StackFrames(env, stack=4)
    env = RewardShapingWrapper(env)
    env = SmoothActionWrapper(env, alpha=0.85, max_steer_change=0.15)
    return env