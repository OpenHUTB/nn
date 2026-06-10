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


class CarlaRewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.max_steer = 0.6
        self.consecutive_lane_invasion = 0
        self.consecutive_collision = 0
        self.total_steps = 0

        # 方向盘转角惩罚相关
        self.last_steer = 0.0
        self.consecutive_large_steer = 0
        self.steer_history = []
        self.max_steer_history = 10

    def step(self, action):
        action[0] = np.clip(action[0], -self.max_steer, self.max_steer)
        current_steer = action[0]

        obs, reward, terminated, truncated, info = self.env.step(action)
        speed = info.get('speed', 0.0)
        collision = info.get('collision', False)
        lane_invasion = info.get('lane_invasion', False)

        self.total_steps += 1

        # 计算原始奖励
        original_reward = reward
        shaped_reward = 0.0

        # ===== 速度奖励 =====
        if speed > 0.5:
            speed_reward = min(speed / 50.0, 0.5)
            shaped_reward += speed_reward

        # ===== 方向盘转角惩罚 =====
        # 1. 大转角惩罚（与速度相关）
        steer_magnitude = abs(current_steer)
        if steer_magnitude > 0.2:
            # 高速时大转向惩罚更严重
            speed_factor = min(speed / 30.0, 1.0)
            turn_penalty = steer_magnitude * 0.2 * speed_factor
            shaped_reward -= turn_penalty

            # 连续大转角惩罚
            self.consecutive_large_steer += 1
            if self.consecutive_large_steer > 3:
                shaped_reward -= 0.08 * (self.consecutive_large_steer - 3)
        else:
            self.consecutive_large_steer = 0

        # 2. 方向盘抖动惩罚（快速来回转向）
        self.steer_history.append(current_steer)
        if len(self.steer_history) > self.max_steer_history:
            self.steer_history.pop(0)

        if len(self.steer_history) > 3:
            steer_changes = np.abs(np.diff(self.steer_history))
            avg_steer_change = np.mean(steer_changes)
            if avg_steer_change > 0.1 and speed > 5.0:
                jitter_penalty = min(avg_steer_change * 0.3, 0.2)
                shaped_reward -= jitter_penalty

        # 3. 极端转向惩罚
        if steer_magnitude > 0.4:
            shaped_reward -= 0.3

        # 更新历史转向
        self.last_steer = current_steer

        # ===== 碰撞检测与惩罚 =====
        if collision:
            shaped_reward -= 10.0
            terminated = True

        # ===== 车道入侵惩罚 =====
        if lane_invasion:
            self.consecutive_lane_invasion += 1
            lane_penalty = 0.5 + self.consecutive_lane_invasion * 0.2
            shaped_reward -= lane_penalty
        else:
            self.consecutive_lane_invasion = 0
            if speed > 10.0 and steer_magnitude < 0.2:
                shaped_reward += 0.15

        # ===== 平滑驾驶奖励 =====
        if steer_magnitude < 0.05 and speed > 10.0 and not lane_invasion:
            shaped_reward += 0.1
        elif steer_magnitude > 0.3 and speed > 20.0:
            shaped_reward -= 0.15

        # ===== 高速平稳驾驶额外奖励 =====
        if speed > 30.0 and steer_magnitude < 0.2 and not lane_invasion:
            shaped_reward += 0.3

        # ===== 低速惩罚（防止龟速行驶） =====
        if speed < 5.0 and not lane_invasion:
            shaped_reward -= 0.05

        total_reward = original_reward + shaped_reward

        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.consecutive_lane_invasion = 0
        self.consecutive_collision = 0
        self.total_steps = 0
        self.last_steer = 0.0
        self.consecutive_large_steer = 0
        self.steer_history = []
        return self.env.reset(**kwargs)


def wrap_env(env):
    env = SkipFrame(env, skip=4)
    env = PreProcessObs(env)
    env = StackFrames(env, stack=4)
    env = CarlaRewardShapingWrapper(env)
    env = SmoothActionWrapper(env, alpha=0.85, max_steer_change=0.15)
    return env
