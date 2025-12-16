from stable_baselines3 import PPO
from custom_env import AirSimMazeEnv
import time


def main():
    # 加载环境 (用于验证)
    env = AirSimMazeEnv()

    # 加载已训练的模型 (请确保文件名对应)
    model_path = "models/drone_maze_final"  # 或者某个检查点文件
    try:
        model = PPO.load(model_path)
        print(f"成功加载模型: {model_path}")
    except FileNotFoundError:
        print("未找到模型文件，请先运行 train.py")
        return

    obs, _ = env.reset()
    print("开始自动寻路测试...")

    done = False
    while True:
        # 模型预测动作 (deterministic=True 表示不使用随机性，完全确定的策略)
        action, _states = model.predict(obs, deterministic=True)

        # 环境执行动作
        obs, reward, done, truncated, info = env.step(action)

        if done:
            print("回合结束 (碰撞或到达终点)")
            time.sleep(1)
            obs, _ = env.reset()


if __name__ == "__main__":
    main()