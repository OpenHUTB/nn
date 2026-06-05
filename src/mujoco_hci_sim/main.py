import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# ===================== 【稳走专用参数】 =====================
ENV_NAME = "Humanoid-v4"
MODEL_DIR = "models_stable"
LOG_DIR = "logs_stable"
TOTAL_TIMESTEPS = 8_000_000    # 训练800万步，足够学稳定
TEST_EPISODES = 20
EVAL_FREQ = 20_000             # 频繁评估，只保留最稳的模型
SAVE_FREQ = 200_000
# ==========================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model")

# 动作平滑：解决关节突然发力导致的摔倒
class ActionSmoothingWrapper(gym.ActionWrapper):
    def __init__(self, env, alpha=0.8):
        super().__init__(env)
        self.alpha = alpha
        self.last_action = np.zeros(env.action_space.shape)

    def action(self, action):
        smoothed = self.alpha * self.last_action + (1 - self.alpha) * action
        self.last_action = smoothed
        return smoothed

# 修复后的自定义奖励包装器
class CustomRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_action = np.zeros(env.action_space.shape)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        data = self.env.unwrapped.data
        # 前进奖励
        forward_reward = data.qvel[0] * 1.2
        # 直立高度惩罚
        upright_penalty = -0.6 * abs(data.qpos[2] - 1.2)
        # 功耗惩罚
        energy_penalty = -1e-3 * np.sum(np.square(data.ctrl))
        # 动作剧烈变化惩罚
        action_smooth_pen = -0.01 * np.linalg.norm(action - self.last_action)
        self.last_action = action.copy()
        # 组合总奖励
        total = forward_reward + upright_penalty + energy_penalty + action_smooth_pen + reward * 0.3
        total = np.clip(total, -5.0, 10.0)
        return obs, total, terminated, truncated, info

def make_env(render_mode="human"):
    env = gym.make(ENV_NAME, render_mode=render_mode)
    env = ActionSmoothingWrapper(env, alpha=0.8)
    env = CustomRewardWrapper(env)
    env = DummyVecEnv([lambda: env])
    return env

def train_model(env):
    print("🚀 开始训练【稳定行走】模型...")

    if os.path.exists(FINAL_MODEL_PATH + ".zip"):
        print("✅ 找到已有模型，继续训练...")
        model = PPO.load(FINAL_MODEL_PATH, env=env, tensorboard_log=LOG_DIR)
    else:
        print("🆕 创建【稳走专用】PPO模型...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            learning_rate=1e-4,
            gamma=0.995,
            n_steps=4096,
            batch_size=1024,
            n_epochs=15,
            clip_range=0.15,
            ent_coef=0.01,
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[512, 256, 128],
                    vf=[512, 256, 128]
                )
            )
        )

    eval_callback = EvalCallback(
        env, best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR, eval_freq=EVAL_FREQ,
        deterministic=True, render=False
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ, save_path=MODEL_DIR, name_prefix="chk"
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        reset_num_timesteps=False
    )

    model.save(FINAL_MODEL_PATH)
    print(f"✅ 训练完成！最稳模型已保存到 {BEST_MODEL_PATH}")
    return model

def test_model(env):
    print("🎮 加载【最稳模型】开始测试...")
    if not os.path.exists(BEST_MODEL_PATH + ".zip"):
        print("❌ 请先训练模型！")
        return

    model = PPO.load(BEST_MODEL_PATH, env=env)
    obs = env.reset()
    total_reward = 0
    episode = 0

    while episode < TEST_EPISODES:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        total_reward += reward[0]

        if done:
            episode += 1
            print(f"🏆 第{episode}轮 总奖励：{total_reward:.2f}")
            total_reward = 0
            obs = env.reset()

    print("✅ 测试完成！")

if __name__ == "__main__":
    try:
        env = make_env(render_mode="human")
        # 训练时启用下面train，测试启用testS
        #train_model(env)
        test_model(env)
    except KeyboardInterrupt:
        print("\n🛑 手动停止训练")
    finally:
        env.close()
        print("👋 环境已关闭")