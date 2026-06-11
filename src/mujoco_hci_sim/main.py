import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import get_linear_fn

# ===================== 配置区 =====================
ENV_NAME = "Humanoid-v4"
MODEL_DIR = "models_stable"
LOG_DIR = "logs_stable"

TOTAL_TIMESTEPS = 8_000_000
TEST_EPISODES = 20
EVAL_FREQ = 2000  # 调小评估频率，更容易触发保存最优模型
SAVE_FREQ = 200_000
OBS_HIST_LEN = 3
ACTION_NOISE_STD = 0.1

BEST_MODEL_ZIP = os.path.join(MODEL_DIR, "best_model.zip")
USE_OBS_HIST = not os.path.exists(BEST_MODEL_ZIP)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model")
VEC_NORMALIZE_PATH = os.path.join(MODEL_DIR, "vec_normalize.pkl")

# ===================== 环境包装器 =====================
class ActionSmoothingWrapper(gym.ActionWrapper):
    def __init__(self, env, alpha=0.8):
        super().__init__(env)
        self.alpha = alpha
        self.last_action = np.zeros(env.action_space.shape)

    def action(self, action):
        smoothed = self.alpha * self.last_action + (1 - self.alpha) * action
        self.last_action = smoothed
        return smoothed

class ActionNoiseWrapper(gym.ActionWrapper):
    def __init__(self, env, noise_std=0.1):
        super().__init__(env)
        self.noise_std = noise_std

    def action(self, action):
        env_ref = self
        while hasattr(env_ref, 'env'):
            env_ref = env_ref.env
        training_mode = getattr(env_ref, 'training', False)

        if training_mode:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        return action

class ObsHistoryWrapper(gym.ObservationWrapper):
    def __init__(self, env, hist_len=3):
        super().__init__(env)
        self.hist_len = hist_len
        self.obs_buffer = []
        obs_dim = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim * hist_len,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.obs_buffer = [obs.copy() for _ in range(self.hist_len)]
        return np.concatenate(self.obs_buffer), info

    def observation(self, obs):
        self.obs_buffer.pop(0)
        self.obs_buffer.append(obs.copy())
        return np.concatenate(self.obs_buffer)

class CustomRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_action = np.zeros(env.action_space.shape)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        data = self.env.unwrapped.data

        forward_reward = data.qvel[0] * 1.2
        upright_penalty = -0.6 * abs(data.qpos[2] - 1.2)
        energy_penalty = -1e-3 * np.sum(np.square(data.ctrl))
        action_smooth_pen = -0.01 * np.linalg.norm(action - self.last_action)
        self.last_action = action.copy()

        contact_penalty = 0.0
        for con in data.contact:
            g1, g2 = con.geom1, con.geom2
            if g1 not in (4,5) and g2 not in (4,5):
                contact_penalty -= 0.08

        total = forward_reward + upright_penalty + energy_penalty + action_smooth_pen + contact_penalty + reward * 0.3
        total = np.clip(total, -5.0, 10.0)
        return obs, total, terminated, truncated, info

# ===================== 环境创建 =====================
def make_env(render_mode=None):
    def _init():
        env = gym.make(ENV_NAME, render_mode=render_mode)
        env = ActionSmoothingWrapper(env, alpha=0.8)
        env = ActionNoiseWrapper(env, noise_std=ACTION_NOISE_STD)
        env = CustomRewardWrapper(env)
        if USE_OBS_HIST:
            env = ObsHistoryWrapper(env, hist_len=OBS_HIST_LEN)
        return env
    return DummyVecEnv([_init])

def make_train_env():
    env = make_env(render_mode=None)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.995)
    return env

def make_test_env(render_mode="human"):
    env = make_env(render_mode=render_mode)
    if os.path.exists(VEC_NORMALIZE_PATH):
        env = VecNormalize.load(VEC_NORMALIZE_PATH, env)
        env.training = False
        env.norm_reward = False
    return env

# ===================== 训练 =====================
def train_model():
    print("🚀 训练 | 历史观测:", "开启" if USE_OBS_HIST else "关闭")
    print("🔊 动作噪声:", ACTION_NOISE_STD)

    env = make_train_env()

    if os.path.exists(FINAL_MODEL_PATH + ".zip"):
        print("✅ 加载已有模型继续训练")
        model = PPO.load(FINAL_MODEL_PATH, env=env, tensorboard_log=LOG_DIR)
    else:
        print("🆕 新建PPO模型")
        lr_schedule = get_linear_fn(1e-4, 1e-5, 1.0)
        model = PPO(
            "MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR,
            learning_rate=lr_schedule,
            gamma=0.995,
            gae_lambda=0.97,
            n_steps=4096,
            batch_size=1024,
            n_epochs=15,
            clip_range=0.15,
            ent_coef=0.008,
            policy_kwargs=dict(net_arch=dict(pi=[512,256,128], vf=[512,256,128]))
        )

    eval_cb = EvalCallback(
        env, best_model_save_path=MODEL_DIR, log_path=LOG_DIR,
        eval_freq=EVAL_FREQ, deterministic=True, render=False
    )
    ckpt_cb = CheckpointCallback(save_freq=SAVE_FREQ, save_path=MODEL_DIR, name_prefix="chk")

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[eval_cb, ckpt_cb], reset_num_timesteps=False)

    model.save(FINAL_MODEL_PATH)
    env.save(VEC_NORMALIZE_PATH)
    env.close()
    print("✅ 训练完成")

# ===================== 测试 =====================
def test_model():
    print("🎮 模型测试 | 历史观测:" + ("开启(新版)" if USE_OBS_HIST else "关闭(旧版)"))
    if not os.path.exists(BEST_MODEL_PATH + ".zip"):
        print("❌ 无训练模型，请先执行训练")
        return

    test_env = make_test_env(render_mode="human")
    model = PPO.load(BEST_MODEL_PATH, env=test_env)
    obs = test_env.reset()
    total_reward = 0
    episode = 0

    while episode < TEST_EPISODES:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        total_reward += reward[0]

        if done:
            episode += 1
            print(f"🏆 第{episode}轮 单轮奖励：{total_reward:.2f}")
            total_reward = 0
            obs = test_env.reset()

    test_env.close()
    print("✅ 全部测试完毕")

# ===================== 主入口 =====================
if __name__ == "__main__":
    try:
        # train_model()
         test_model()
    except KeyboardInterrupt:
        print("\n🛑 手动停止")