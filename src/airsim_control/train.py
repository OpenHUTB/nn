from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from custom_env import AirSimMazeEnv
import os

# 创建日志目录
log_dir = "./airsim_logs/"
os.makedirs(log_dir, exist_ok=True)


def main():
    # 实例化环境
    env = AirSimMazeEnv()

    # 定义模型
    # MultiInputPolicy 会自动检测 Dict 输入
    # learning_rate: 学习率，通常 3e-4 是个不错的起点
    # n_steps: 每次更新收集的步数
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=0.0003,
        batch_size=64,
        n_steps=2048,
        gamma=0.99
    )

    # 保存检查点的回调函数 (每 10000 步保存一次)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models/',
        name_prefix='drone_maze'
    )

    print("开始训练...")
    # 训练步数建议：简单的迷宫可能需要 20万-50万步，复杂的需要更多
    model.learn(total_timesteps=500000, callback=checkpoint_callback)

    # 保存最终模型
    model.save("drone_maze_final")
    print("训练完成，模型已保存。")


if __name__ == "__main__":
    main()