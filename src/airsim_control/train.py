import glob
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from custom_env import AirSimMazeEnv

# === 配置路径 (使用你提供的绝对路径) ===
MODELS_DIR = r"D:\Others\MyAirsimprojects\models"
LOG_DIR = r"D:\Others\MyAirsimprojects\airsim_logs"

# 确保目录存在
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def get_latest_model_path(path_dir):
    """
    辅助函数：查找指定目录下最新的 .zip 模型文件
    """
    # 获取目录下所有 .zip 文件
    list_of_files = glob.glob(os.path.join(path_dir, '*.zip'))

    if not list_of_files:
        return None

    # 按照文件修改时间排序，取最新的一个
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def main():
    # 实例化环境
    env = AirSimMazeEnv()

    # 1. 尝试寻找最新的模型
    latest_model_path = get_latest_model_path(MODELS_DIR)

    if latest_model_path:
        print(f"--------------------------------------------------")
        print(f"检测到已存在的模型，正在加载: {latest_model_path}")
        print(f"将恢复之前的训练进度...")
        print(f"--------------------------------------------------")

        # === 加载旧模型 ===
        # 注意：这里传入 env 和 tensorboard_log 是为了确保训练环境和日志路径正确
        model = PPO.load(latest_model_path, env=env, tensorboard_log=LOG_DIR)

        # 标记：不需要重置步数，我们要接着画图
        reset_timesteps = False

    else:
        print(f"--------------------------------------------------")
        print(f"未找到已有模型，正在初始化新模型 (From Scratch)...")
        print(f"--------------------------------------------------")

        # === 创建新模型 ===
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            learning_rate=0.0003,
            batch_size=64,
            n_steps=2048,
            gamma=0.99
        )
        # 标记：新模型需要从 0 开始计数
        reset_timesteps = True

    # 定义回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=MODELS_DIR,  # 使用你指定的 models 路径
        name_prefix='drone_maze'
    )

    print("开始/继续 训练...")

    # === 开始训练 ===
    # reset_num_timesteps=False 关键！
    # 如果是加载的模型，设为 False 可以让 Tensorboard 的曲线接上之前的，而不是从 0 开始乱画。
    model.learn(
        total_timesteps=500000,
        callback=checkpoint_callback,
        reset_num_timesteps=reset_timesteps
    )

    # 保存最终模型
    final_save_path = os.path.join(MODELS_DIR, "drone_maze_final")
    model.save(final_save_path)
    print(f"训练完成，最终模型已保存至: {final_save_path}")


if __name__ == "__main__":
    main()