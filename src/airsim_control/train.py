import glob
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from custom_env import AirSimMazeEnv

# === 路径配置 ===
MODELS_DIR = r"D:\Others\MyAirsimprojects\models"
LOG_DIR = r"D:\Others\MyAirsimprojects\airsim_logs"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def get_latest_model_path(path_dir):
    list_of_files = glob.glob(os.path.join(path_dir, '*.zip'))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)


def main():
    env = AirSimMazeEnv()
    latest_model_path = get_latest_model_path(MODELS_DIR)

    if latest_model_path:
        print(f"--- 发现存档: {latest_model_path}，继续训练 ---")
        model = PPO.load(latest_model_path, env=env, tensorboard_log=LOG_DIR)
        reset_timesteps = False
    else:
        print(f"--- 未发现存档，开始【从头训练】 ---")
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            learning_rate=0.0003,
            batch_size=256,  # 大Batch加速
            n_steps=2048,
            gamma=0.99
        )
        reset_timesteps = True

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=MODELS_DIR,
        name_prefix='drone_maze'
    )

    print("🚀 训练引擎启动...")
    model.learn(
        total_timesteps=500000,
        callback=checkpoint_callback,
        reset_num_timesteps=reset_timesteps
    )

    model.save(os.path.join(MODELS_DIR, "drone_maze_final"))
    print("训练结束。")


if __name__ == "__main__":
    main()