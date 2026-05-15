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
    def __init__(self, env, alpha=0.75, max_steer_change=0.3):
        super().__init__(env)
        self.alpha = alpha
        self.max_steer_change = max_steer_change
        self.last_action = None

    def step(self, action):
        if self.last_action is not None:
            # 根据当前转向角度调整平滑强度
            if abs(action[0]) > 0.2:
                # 大转向时减少平滑
                smooth_factor = 0.65
            else:
                smooth_factor = self.alpha

            action = smooth_factor * action + (1 - smooth_factor) * self.last_action
            action[0] = np.clip(action[0],
                                self.last_action[0] - self.max_steer_change,
                                self.last_action[0] + self.max_steer_change)
        self.last_action = action.copy()
        return self.env.step(action)

    def reset(self, **kwargs):
        self.last_action = None
        return self.env.reset(**kwargs)

class TrackDetectionWrapper(gym.Wrapper):
    """基于图像检测车辆是否在赛道上的包装器"""
    def __init__(self, env):
        super().__init__(env)
        # 赛道颜色阈值（RGB）
        # 赛道通常是深灰色，路边有红色/绿色标记
        self.track_low = np.array([90, 90, 90], dtype=np.uint8)
        self.track_high = np.array([120, 120, 120], dtype=np.uint8)
        self.grass_low1 = np.array([0, 100, 0], dtype=np.uint8)
        self.grass_high1 = np.array([80, 180, 80], dtype=np.uint8)
        self.grass_low2 = np.array([0, 140, 0], dtype=np.uint8)
        self.grass_high2 = np.array([100, 255, 100], dtype=np.uint8)

    def _check_on_track(self, obs):
        """检测车辆是否在赛道上"""
        # 检查图像底部中心区域（车辆位置）
        h, w = obs.shape[:2]
        check_region = obs[int(h*0.65):int(h*0.85), int(w*0.35):int(w*0.65)]

        # 检测赛道像素（灰色）
        track_mask = cv2.inRange(check_region, self.track_low, self.track_high)

        # 检测草地像素
        grass_mask1 = cv2.inRange(check_region, self.grass_low1, self.grass_high1)
        grass_mask2 = cv2.inRange(check_region, self.grass_low2, self.grass_high2)
        grass_mask = cv2.bitwise_or(grass_mask1, grass_mask2)

        track_ratio = np.sum(track_mask > 0) / (check_region.shape[0] * check_region.shape[1])
        grass_ratio = np.sum(grass_mask > 0) / (check_region.shape[0] * check_region.shape[1])

        # 如果赛道像素比例足够高，认为在赛道上
        on_track = track_ratio > 0.15
        on_grass = grass_ratio > 0.3

        return on_track, on_grass, track_ratio, grass_ratio

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 检测是否在赛道上
        on_track, on_grass, track_ratio, grass_ratio = self._check_on_track(obs)

        # 更新 info
        info['on_track'] = on_track
        info['on_grass'] = on_grass
        info['track_ratio'] = track_ratio
        info['grass_ratio'] = grass_ratio

        return obs, reward, terminated, truncated, info


class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.max_steer = 0.8  # 增加最大转向角度，允许更大的转弯
        self.consecutive_off_track = 0
        self.consecutive_on_grass = 0
        self.max_off_track_steps = 10  # 稍微延长出赛道容忍时间
        self.max_grass_steps = 18  # 稍微延长草地行驶容忍时间
        self.last_on_track_step = 0
        self.total_steps = 0

        # 方向盘转角惩罚相关
        self.last_steer = 0.0
        self.consecutive_large_steer = 0
        self.steer_history = []
        self.max_steer_history = 10

        # 跟踪车辆状态
        self.last_speed = 0.0
        self.cornering_balance = 0.0

    def step(self, action):
        action[0] = np.clip(action[0], -self.max_steer, self.max_steer)
        current_steer = action[0]

        obs, reward, terminated, truncated, info = self.env.step(action)
        speed = info.get('speed', 0.0)
        on_track = info.get('on_track', True)
        on_grass = info.get('on_grass', False)
        track_ratio = info.get('track_ratio', 0.5)

        self.total_steps += 1

        # 计算原始奖励
        original_reward = reward
        shaped_reward = 0.0

        # ===== 速度奖励 =====
        if speed > 0.5:
            # 根据转向角度调整理想速度
            steer_magnitude = abs(current_steer)
            # 转弯时应该减速，直线时应该加速
            ideal_speed = max(2.0 - steer_magnitude * 3.0, 1.0)
            speed_diff = abs(speed - ideal_speed)
            speed_reward = max(0.0, 0.5 - speed_diff * 0.2)
            shaped_reward += speed_reward

            # 基础前进奖励
            progress_reward = min(speed / 15.0, 0.3)
            shaped_reward += progress_reward

        # ===== 方向盘转角奖励/惩罚 =====
        steer_magnitude = abs(current_steer)

        # 1. 适度转弯奖励（不惩罚正常转弯）
        if steer_magnitude > 0.1 and steer_magnitude < 0.6:
            # 有转向但不过度，给予小奖励
            turn_reward = 0.05 * steer_magnitude
            shaped_reward += turn_reward

        # 2. 转弯减速配合奖励
        if steer_magnitude > 0.2 and self.last_speed > speed and speed > 2.0:
            # 转弯时适当减速是好的
            cornering_reward = 0.1
            shaped_reward += cornering_reward

        # 3. 大转角惩罚（只在高速时）
        if steer_magnitude > 0.6:
            speed_factor = min(speed / 8.0, 1.0)
            turn_penalty = steer_magnitude * 0.15 * speed_factor
            shaped_reward -= turn_penalty

            self.consecutive_large_steer += 1
            if self.consecutive_large_steer > 5:
                shaped_reward -= 0.05 * (self.consecutive_large_steer - 5)
        else:
            self.consecutive_large_steer = 0

        # 4. 方向盘抖动惩罚（快速来回转向）
        self.steer_history.append(current_steer)
        if len(self.steer_history) > self.max_steer_history:
            self.steer_history.pop(0)

        if len(self.steer_history) > 4:
            steer_changes = np.abs(np.diff(self.steer_history))
            avg_steer_change = np.mean(steer_changes)
            if avg_steer_change > 0.15 and speed > 3.0:
                jitter_penalty = min(avg_steer_change * 0.2, 0.15)
                shaped_reward -= jitter_penalty

        # 5. 极端转向惩罚
        if steer_magnitude > 0.7:
            shaped_reward -= 0.2

        # 更新历史转向和速度
        self.last_steer = current_steer
        self.last_speed = speed

        # ===== 基于图像的出赛道检测 =====
        if not on_track or original_reward < -0.5:
            self.consecutive_off_track += 1
            off_track_penalty = 1.5 + self.consecutive_off_track * 0.8
            shaped_reward -= off_track_penalty

            # 记录最后一次在赛道上的时间步
            if self.consecutive_off_track == 1:
                self.last_on_track_step = self.total_steps - 1

            if self.consecutive_off_track >= self.max_off_track_steps:
                truncated = True
                shaped_reward -= 10.0
        else:
            self.consecutive_off_track = 0
            # 在赛道上的奖励
            if original_reward > -0.1:
                shaped_reward += 0.15
            # 高赛道比例额外奖励
            if track_ratio > 0.4:
                shaped_reward += 0.1

        # ===== 草地检测与惩罚 =====
        if on_grass:
            self.consecutive_on_grass += 1
            grass_penalty = 0.3 + self.consecutive_on_grass * 0.2
            shaped_reward -= grass_penalty

            # 在草地上速度越高惩罚越大
            if speed > 1.0:
                shaped_reward -= speed * 0.2

            if self.consecutive_on_grass >= self.max_grass_steps:
                truncated = True
                shaped_reward -= 5.0
        else:
            self.consecutive_on_grass = 0

        # ===== 平滑驾驶奖励 =====
        steer_magnitude = abs(action[0])
        if steer_magnitude < 0.05 and speed > 1.0 and on_track:
            shaped_reward += 0.05
        elif steer_magnitude > 0.3 and speed > 2.0:
            shaped_reward -= 0.1

        # ===== 高速平稳驾驶额外奖励 =====
        if speed > 2.0 and steer_magnitude < 0.2 and on_track and original_reward > -0.1:
            shaped_reward += 0.25

        # ===== 低速惩罚（防止龟速行驶） =====
        if speed < 0.3 and on_track and original_reward > -0.1:
            shaped_reward -= 0.05

        # ===== 回到赛道的奖励 =====
        if self.consecutive_off_track == 0 and self.total_steps - self.last_on_track_step <= 10 and self.last_on_track_step > 0:
            shaped_reward += 2.0
            self.last_on_track_step = 0

        total_reward = original_reward + shaped_reward

        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.consecutive_off_track = 0
        self.consecutive_on_grass = 0
        self.last_on_track_step = 0
        self.total_steps = 0
        self.last_steer = 0.0
        self.consecutive_large_steer = 0
        self.steer_history = []
        return self.env.reset(**kwargs)

def wrap_env(env):
    env = SkipFrame(env, skip=4)
    env = TrackDetectionWrapper(env)  # 在预处理前使用原始图像检测赛道
    env = PreProcessObs(env)
    env = StackFrames(env, stack=4)
    env = RewardShapingWrapper(env)
    env = SmoothActionWrapper(env, alpha=0.75, max_steer_change=0.3)
    return env