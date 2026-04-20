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


class AntiSpinWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.last_pos = None  # 记录上一帧位置
        self.turn_penalty = 0.1  # 转向惩罚系数
        self.stagnation_penalty = 0.5  # 原地惩罚系数
        self.speed_reward = 0.2  # 前进奖励系数
        self.max_steer_angle = 0.8  # 限制最大转向角

    def step(self, action):
        # 1. 限制最大转向角，防止极端转向
        action[0] = np.clip(action[0], -self.max_steer_angle, self.max_steer_angle)

        # 2. 执行动作并获取原始反馈
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 3. 获取车辆速度和位置（CarRacing-v3的info包含速度信息）
        speed = info.get('speed', 0.0)
        pos = info.get('position', (0, 0))

        # 4. 计算转向惩罚（转向角越大+速度越低，惩罚越重）
        steer_penalty = abs(action[0]) * (1 - min(1.0, abs(speed))) * self.turn_penalty

        # 5. 计算原地惩罚（位置变化小+速度低）
        stagnation_penalty = 0.0
        if self.last_pos is not None:
            pos_diff = np.linalg.norm(np.array(pos) - np.array(self.last_pos))
            if pos_diff < 0.1 and abs(speed) < 0.5:
                stagnation_penalty = self.stagnation_penalty

        # 6. 前进奖励（正向速度奖励）
        forward_reward = max(0.0, speed) * self.speed_reward

        # 7. 综合奖励（原始奖励 + 前进奖励 - 转向惩罚 - 原地惩罚）
        reward = reward + forward_reward - steer_penalty - stagnation_penalty

        # 8. 更新位置记录
        self.last_pos = pos

        # 9. 极端情况：持续原地转圈直接终止回合
        if abs(speed) < 0.1 and abs(action[0]) > 0.5 and self.last_pos is not None:
            truncated = True  # 触发截断，结束当前回合

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.last_pos = None
        return self.env.reset(**kwargs)


def wrap_env(env):
    env = SkipFrame(env)
    env = PreProcessObs(env)
    env = StackFrames(env)
    env = SmoothActionWrapper(env, alpha=0.7)  # 添加动作平滑
    env = AntiSpinWrapper(env)  # 添加防原地转圈约束
    return env