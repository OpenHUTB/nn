from stable_baselines3 import PPO
from custom_env import AirSimMazeEnv
import time
import os

MODEL_PATH = "models/drone_maze_final"  # 或者具体 checkpoint zip
# 如果你有多个 final 文件，用绝对路径或最近的模型路径替换上面

def main():
    env = AirSimMazeEnv()
    model = None
    try:
        # 尝试加载模型
        if os.path.exists(MODEL_PATH + ".zip"):
            model = PPO.load(MODEL_PATH, env=env)
            print("成功加载模型:", MODEL_PATH)
        else:
            # 尝试寻找目录下最新 zip
            import glob
            zl = glob.glob("models/*.zip")
            if zl:
                model = PPO.load(sorted(zl, key=os.path.getctime)[-1], env=env)
                print("加载最新模型:", sorted(zl, key=os.path.getctime)[-1])
            else:
                print("未找到模型，请先训练并保存模型。")
                return
    except Exception as e:
        print("加载模型失败:", e)
        return

    obs, _ = env.reset()
    print("开始推理测试（按 Ctrl+C 停止）")

    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            # 简单日志
            print(f"step reward={reward:.3f}, done={done}")
            if done:
                print("回合结束，reset...")
                time.sleep(0.5)
                obs, _ = env.reset()
            # 小延迟以便观察（可移除）
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("推理停止")
    finally:
        env.close()

if __name__ == "__main__":
    main()
