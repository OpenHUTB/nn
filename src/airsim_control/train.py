import os
import glob
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from custom_env import AirSimMazeEnv

# === 配置路径（根据你的环境修改） ===
MODELS_DIR = r"D:\Others\MyAirsimprojects\models"
LOG_DIR = r"D:\Others\MyAirsimprojects\airsim_logs"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def get_latest_model_path(path_dir):
    list_of_files = glob.glob(os.path.join(path_dir, '*.zip'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def make_env(include_velocity=True):
    def _init():
        env = AirSimMazeEnv(include_velocity=include_velocity)
        env = Monitor(env)
        return env
    return _init

def main():
    latest_model_path = get_latest_model_path(MODELS_DIR)
    reset_timesteps = True
    include_velocity = True  # 默认使用 velocity

    # 如果存在已保存的模型，先加载模型对象（不传 env）以检查 observation_space 是否包含 velocity
    if latest_model_path:
        print("检测到模型，先临时加载模型对象以检查 observation_space:", latest_model_path)
        try:
            tmp_model = PPO.load(latest_model_path, env=None)  # 不传 env，不触发空间检查
            # tmp_model.observation_space 是保存时的 observation_space
            saved_obs_space = getattr(tmp_model, "observation_space", None)
            if saved_obs_space is not None and isinstance(saved_obs_space, dict) is False:
                # 在某些 SB3 版本中 observation_space 是 spaces.Dict 对象，检查其 spaces 字段
                try:
                    saved_keys = list(saved_obs_space.spaces.keys())
                except Exception:
                    saved_keys = []
            else:
                # 保险 fallback
                try:
                    saved_keys = list(saved_obs_space.spaces.keys())
                except Exception:
                    saved_keys = []
            print("已保存模型的 observation keys:", saved_keys)
            include_velocity = ("velocity" in saved_keys)
            print(f"将创建 include_velocity={include_velocity} 的环境以兼容模型。")
            reset_timesteps = False
        except Exception as e:
            print("尝试读取已保存模型的 observation_space 失败，将按默认创建新环境。错误：", e)
            latest_model_path = None  # 退回为新模型流程

    # 创建向量化环境（单副本），根据 saved model 决定是否包含 velocity
    vec_env = DummyVecEnv([make_env(include_velocity=include_velocity)])

    if latest_model_path:
        # 现在用兼容的 env 真的加载模型并继续训练
        print("正在用兼容的 env 加载并恢复模型：", latest_model_path)
        model = PPO.load(latest_model_path, env=vec_env, tensorboard_log=LOG_DIR)
    else:
        print("未检测到兼容模型，初始化新模型...")
        model = PPO(
            policy="MultiInputPolicy",
            env=vec_env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            learning_rate=3e-4,
            batch_size=64,
            n_steps=2048,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2
        )

    # Checkpoint 回调
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=MODELS_DIR,
        name_prefix='drone_maze'
    )

    print("开始训练...")
    total_timesteps = 500_000
    try:
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, reset_num_timesteps=reset_timesteps)
    except KeyboardInterrupt:
        print("训练被中断，保存临时模型...")
    finally:
        timestamp = int(time.time())
        final_path = os.path.join(MODELS_DIR, f"drone_maze_final_{timestamp}")
        model.save(final_path)
        print("训练结束，模型已保存至:", final_path)

if __name__ == "__main__":
    main()
