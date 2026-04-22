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
        # 返回形状 (stack, h, w) 与 CNN 期望的输入格式一致
        state = np.concatenate(self.frames, axis=-1)
        state = state.transpose(2, 0, 1)
        return state


class SmoothActionWrapper(gym.Wrapper):
    """动作平滑包装器 - 让车辆运行更平滑"""

    def __init__(self, env, alpha=0.85):
        super().__init__(env)
        self.alpha = alpha
        self.last_action = None

    def step(self, action):
        if self.last_action is not None:
            # 指数移动平均平滑
            action = self.alpha * action + (1 - self.alpha) * self.last_action

            # 额外的转向动作约束：限制转向变化率
            action[0] = np.clip(action[0],
                                self.last_action[0] - 0.1,
                                self.last_action[0] + 0.1)

        self.last_action = action.copy()
        return self.env.step(action)

    def reset(self, **kwargs):
        self.last_action = None
        return self.env.reset(**kwargs)


class AntiSpinWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_pos = None
        self.turn_penalty = 0.2
        self.stagnation_penalty = 1.0
        self.speed_reward = 0.3
        self.max_steer_angle = 0.5
        self.spin_threshold = 0.15
        self.position_history = []

    def step(self, action):
        # 1. 严格限制最大转向角
        action[0] = np.clip(action[0], -self.max_steer_angle, self.max_steer_angle)

        # 2. 执行动作
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 3. 获取速度（CarRacing-v3 使用 'speed' 键）
        speed = info.get('speed', 0.0)
        pos = info.get('position', (0, 0))

        # 4. 位置历史
        self.position_history.append(pos)
        if len(self.position_history) > 10:
            self.position_history.pop(0)

        # 5. 转向惩罚
        steer_penalty = abs(action[0]) * (1 - min(1.0, abs(speed))) * self.turn_penalty

        # 6. 原地惩罚
        stagnation_penalty = 0.0
        pos_range = np.array([0, 0])
        if len(self.position_history) >= 10:
            positions = np.array(self.position_history)
            pos_range = np.max(positions, axis=0) - np.min(positions, axis=0)
            if np.all(pos_range < 0.2) and abs(speed) < self.spin_threshold:
                stagnation_penalty = self.stagnation_penalty

        # 7. 速度奖励
        forward_reward = max(0.0, speed) * self.speed_reward
        backward_penalty = min(0.0, speed) * 0.5
        speed_reward = forward_reward + backward_penalty

        # 8. 综合奖励
        reward = reward + speed_reward - steer_penalty - stagnation_penalty

        # 9. 更新位置
        self.last_pos = pos

        # 10. 极端情况：持续原地转圈直接终止
        if (len(self.position_history) >= 10 and
                np.all(pos_range < 0.2) and
                abs(speed) < self.spin_threshold and
                abs(action[0]) > 0.2):
            truncated = True
            reward -= 5.0

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.last_pos = None
        self.position_history = []
        return self.env.reset(**kwargs)

def wrap_env(env):
    """应用所有包装器到环境"""
    env = SkipFrame(env, skip=4)
    env = PreProcessObs(env)
    env = StackFrames(env, stack=4)
    env = AntiSpinWrapper(env)
    env = SmoothActionWrapper(env, alpha=0.85)
    return env