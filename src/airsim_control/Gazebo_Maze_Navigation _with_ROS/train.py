import os
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from custom_env import AirSimMazeEnv  # 导入刚才那个文件

# === 路径配置 (已改为相对路径) ===
# 获取当前脚本文件所在的绝对目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 拼接路径：在脚本同级目录下生成 models 和 logs
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def main():
    # 实例化环境
    env = DummyVecEnv([lambda: AirSimMazeEnv()])

    # === 网络架构配置 ===
    # 因为输入是雷达数据(一维数值)，所以使用 MlpPolicy (多层感知机)
    policy_kwargs = dict(
        activation_fn=th.nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    print("🚀 开始训练 (ROS 2 版)...")
    print(f"数据保存路径: {SCRIPT_DIR}")

    # 初始化 PPO 模型
    model = PPO(
        "MlpPolicy",  # 关键点：雷达数据必须用 MlpPolicy
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=0.0003,
        batch_size=256,
        n_steps=2048,
        gamma=0.99,
        policy_kwargs=policy_kwargs,
        device="auto"
    )

    # 自动保存回调 (每 10000 步保存一次)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=MODELS_DIR,
        name_prefix='ros_drone'
    )

    # 开始学习 (训练 10万步试试)
    model.learn(
        total_timesteps=100000,
        callback=checkpoint_callback
    )

    # 保存最终模型
    model.save(os.path.join(MODELS_DIR, "ros_drone_final"))
    print("训练结束。")


if __name__ == "__main__":
    main()