# eval_agent.py
import os
import sys
import numpy as np

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from carla_env.carla_env_multi_obs import CarlaEnvMultiObs


def main():
    model_path = "final_model.zip"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return

    print("🔄 加载模型...")
    model = PPO.load(model_path)
    print("✅ 模型加载成功！")

    # ✅ 关键：启用 keep_alive_after_exit=True（默认已为 True）
    env = CarlaEnvMultiObs(keep_alive_after_exit=True)

    try:
        obs, _ = env.reset()
        print("▶️ 开始驾驶演示（运行 200 步）...")

        for step in range(200):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if step % 50 == 0:
                x, y, vx, vy = obs
                speed = np.linalg.norm([vx, vy])
                print(f" Step {step}: 位置=({x:.1f}, {y:.1f}), 速度={speed:.2f} m/s")

        print("✅ 演示完成！")

    except KeyboardInterrupt:
        print("\n🛑 用户中断")
    except Exception as e:
        print(f"💥 运行时错误: {e}")
        raise
    finally:
        # 注意：env.close() 会松开控制，但不会销毁车辆（因为 keep_alive=True）
        env.close()

        # ✅✅✅ 关键新增：阻塞进程，防止退出 → 车辆保留在 CARLA 中
        print("\n" + "="*60)
        print("🚗 车辆已保留在 CARLA 中！")
        print("💡 操作指南：")
        print("   1. 切换到 CARLA 窗口")
        print("   2. 按 F5 键进入第三人称跟随视角（推荐）")
        print("   3. 可自由旋转/平移视角观察车辆")
        print("   4. 录制 GIF 或截图")
        print("\n🛑 准备好后，请回到本窗口按 Enter 键退出...")
        input()  # ⬅️ 阻塞：只要不按回车，Python 进程就不退出，车就不会消失
        print("👋 再见！")


if __name__ == "__main__":
    main()