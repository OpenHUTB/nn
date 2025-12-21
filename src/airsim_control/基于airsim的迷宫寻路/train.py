import glob
import os
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from custom_env import AirSimMazeEnv
from typing import Callable

# === 路径配置 ===
MODELS_DIR = r"D:\Others\MyAirsimprojects\models"
LOG_DIR = r"D:\Others\MyAirsimprojects\airsim_logs"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# === 优化 1: 线性学习率调度器 ===
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    线性学习率调度函数。
    :param initial_value: 初始学习率
    :return: 当前步数的学习率
    """

    def func(progress_remaining: float) -> float:
        # progress_remaining 会从 1.0 (开始) 线性减少到 0.0 (结束)
        return progress_remaining * initial_value

    return func


def get_latest_model_path(path_dir):
    list_of_files = glob.glob(os.path.join(path_dir, '*.zip'))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)


def main():
    set_random_seed(42)  # 固定随机种子，方便复现

    # 实例化环境
    # 使用 DummyVecEnv 包装，虽然只有一个环境，但这是标准做法，方便未来扩展
    env = DummyVecEnv([lambda: AirSimMazeEnv()])
    env = VecMonitor(env)  # 添加 Monitor 以记录详细日志

    latest_model_path = get_latest_model_path(MODELS_DIR)

    # === 优化 2: 自定义网络架构 ===
    # net_arch=[256, 256]: 使用两个 256 神经元的隐藏层 (比默认的 64 强大很多)
    # activation_fn: 使用 Tanh 或 ReLU
    policy_kwargs = dict(
        activation_fn=th.nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    if latest_model_path:
        print(f"--- 发现存档: {latest_model_path}，继续训练 ---")
        # 加载旧模型
        # 注意：加载旧模型时，learning_rate 会被覆盖为旧模型的设置
        # 如果你想在旧模型上强制使用新参数，需要手动修改 model.learning_rate
        model = PPO.load(latest_model_path, env=env, tensorboard_log=LOG_DIR)

        # 强制更新学习率策略 (可选，如果你想让旧模型也享受线性衰减)
        # model.lr_schedule = linear_schedule(0.0003)

        reset_timesteps = False
    else:
        print(f"--- 未发现存档，开始【从头训练】(优化版) ---")
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,

            # --- 核心超参数优化 ---
            learning_rate=linear_schedule(0.0003),  # 动态学习率
            batch_size=256,  # 大批量，更稳
            n_steps=2048,  # 每次更新前的采样步数
            gamma=0.99,  # 折扣因子 (长远利益)
            gae_lambda=0.95,  # 优势估计
            clip_range=0.2,  # PPO 修剪范围
            ent_coef=0.01,  # 优化 3: 熵系数，鼓励探索，防止过早死板
            policy_kwargs=policy_kwargs,  # 应用更大的网络
            device="auto"  # 自动使用 GPU
        )
        reset_timesteps = True

    checkpoint_callback = CheckpointCallback(
        save_freq=20000,  # 每 2万步保存一次，不用太频繁
        save_path=MODELS_DIR,
        name_prefix='drone_maze_opt'  # 改个名字区分一下
    )

    print("🚀 优化版训练引擎启动...")
    print("配置: Linear LR, Net=[256,256], Ent=0.01")

    model.learn(
        total_timesteps=1000000,  # 建议跑 100万步
        callback=checkpoint_callback,
        reset_num_timesteps=reset_timesteps
    )

    model.save(os.path.join(MODELS_DIR, "drone_maze_final_opt"))
    print("训练结束。")


if __name__ == "__main__":
    main()