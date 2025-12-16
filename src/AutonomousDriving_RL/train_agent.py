# train_agent.py
"""CARLA 强化学习训练脚本，使用 CarlaEnvMultiObs 环境 + PPO 算法"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from carla_env.carla_env_multi_obs import CarlaEnvMultiObs

def main():
    print("🚀 正在创建 CARLA 环境...")
    env = CarlaEnvMultiObs()

    print("🔍 检查环境...")
    check_env(env, warn=True)

    print("🧠 初始化 PPO 模型...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./logs/"
    )

    print("▶️ 开始训练...")
    try:
        # 注意：progress_bar=False 避免 tqdm/rich 依赖错误
        model.learn(total_timesteps=50000, progress_bar=False)
        print("✅ 训练完成！保存模型...")
        model.save("carla_ppo_agent")
    except KeyboardInterrupt:
        print("⚠️ 训练被用户中断")
    finally:
        env.close()
        print("CloseOperation: 环境已关闭")

if __name__ == "__main__":
    main()