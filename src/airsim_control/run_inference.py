import glob
import os
import time
from stable_baselines3 import PPO
from custom_env import AirSimMazeEnv

# === 配置路径 ===
MODELS_DIR = r"D:\Others\MyAirsimprojects\models"


def get_latest_model_path(path_dir):
    list_of_files = glob.glob(os.path.join(path_dir, '*.zip'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def main():
    # 实例化环境
    env = AirSimMazeEnv()

    # 寻找最新的模型
    model_path = get_latest_model_path(MODELS_DIR)

    if not model_path:
        print("错误：没有找到任何训练好的模型！请先运行 train.py。")
        return

    print(f"正在加载模型进行测试: {model_path}")
    model = PPO.load(model_path)

    # 开始测试循环
    obs, _ = env.reset()

    print("开始推理 (按 Ctrl+C 停止)...")
    try:
        while True:
            # 预测动作 (deterministic=True 表示不使用随机探索，只用这一刻认为最好的动作)
            action, _states = model.predict(obs, deterministic=True)

            # 执行动作
            obs, reward, done, truncated, info = env.step(action)

            # 如果结束了 (撞墙或到达)，自动重置
            if done:
                print("--- 回合结束，重置环境 ---")
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("停止测试")
        env.close()


if __name__ == "__main__":
    main()