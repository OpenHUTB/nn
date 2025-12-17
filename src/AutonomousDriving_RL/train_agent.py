# train_agent.py
# 本脚本用于训练基于 PPO 算法的强化学习智能体（Agent），环境为 CARLA 自动驾驶仿真平台。
# 使用 Stable Baselines3 库实现，支持断点续训、自动保存检查点、安全中断等功能。

import os
import sys

# 将当前脚本所在目录添加到 Python 模块搜索路径，确保能正确导入本地模块（如 carla_env）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入 Stable Baselines3 中的 PPO（Proximal Policy Optimization）算法
from stable_baselines3 import PPO

# 导入环境检查工具，用于验证自定义环境是否符合 Gym 接口规范
from stable_baselines3.common.env_checker import check_env

# 导入回调函数：CheckpointCallback，用于定期保存模型检查点
from stable_baselines3.common.callbacks import CheckpointCallback

# 导入自定义的 CARLA 多观测空间环境（包含图像、速度、位置等多种状态信息）
from carla_env.carla_env_multi_obs import CarlaEnvMultiObs


def main():
    """
    主函数：初始化环境、加载/创建模型、启动训练流程。
    """
    print("🔄 初始化 CARLA 环境...")

    # 创建自定义 CARLA 环境实例
    env = CarlaEnvMultiObs()

    try:
        # 使用 Stable Baselines3 提供的 check_env 工具验证环境是否符合 Gym 标准
        # 若不符合，会抛出警告或异常，帮助开发者快速定位问题
        check_env(env, warn=True)
        print("✅ 环境检查通过！")
    except Exception as e:
        # 如果环境检查失败，打印错误信息并安全关闭环境，退出程序
        print(f"❌ 环境检查失败: {e}")
        env.close()
        return

    # 设置回调函数：每训练 10,000 步自动保存一次模型检查点
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # 保存频率（以环境步数为单位）
        save_path="./checkpoints/",  # 检查点保存目录
        name_prefix="ppo_carla"  # 检查点文件名前缀，如 ppo_carla_10000_steps.zip
    )

    # 定义最新模型的路径，用于判断是否需要继续训练
    model_path = "./checkpoints/ppo_carla_latest.zip"

    # 检查是否存在已保存的模型（用于断点续训）
    if os.path.exists(model_path):
        print(f"🔁 加载已有模型: {model_path}")
        # 从指定路径加载预训练模型，并绑定当前环境
        model = PPO.load(model_path, env=env)
        total_timesteps = 100000  # 总训练步数目标（累计）
        reset_num_timesteps = False  # 不重置步数计数器，继续之前的训练进度
    else:
        print("🆕 训练新模型")
        # 创建全新的 PPO 模型
        model = PPO(
            "MlpPolicy",  # 使用全连接神经网络策略（适用于非图像输入）
            env,  # 绑定训练环境
            verbose=1,  # 输出训练日志（1 表示基本信息）
            learning_rate=3e-4,  # 学习率，常用值，平衡收敛速度与稳定性
            n_steps=2048,  # 每次更新策略前收集的环境交互步数（影响样本效率）
            batch_size=64,  # 每次梯度更新使用的样本批次大小
            n_epochs=10,  # 每批数据重复训练的轮数（提升数据利用率）
            tensorboard_log="./logs/"  # TensorBoard 日志目录，用于可视化训练过程
        )
        total_timesteps = 100000  # 总训练步数
        reset_num_timesteps = True  # 重置步数计数器（因为是新训练）

    print("▶️ 开始训练（按 Ctrl+C 可安全中断）...")

    try:
        # 启动模型训练
        model.learn(
            total_timesteps=total_timesteps,  # 总训练步数
            callback=checkpoint_callback,  # 注册回调函数（自动保存）
            reset_num_timesteps=reset_num_timesteps,  # 是否重置内部步数计数
            progress_bar=False  # 不显示进度条（可设为 True 查看进度）
        )
        # 训练正常完成后，保存最终模型
        model.save("final_model")
        print("🎉 训练完成！模型已保存为 final_model.zip")
    except KeyboardInterrupt:
        # 捕获用户中断信号（Ctrl+C），安全保存当前模型
        print("⚠️ 训练被用户中断，正在保存最新模型...")
        model.save("./checkpoints/ppo_carla_latest")
        print("💾 已保存至 ./checkpoints/ppo_carla_latest.zip")
    finally:
        # 无论训练成功与否，都确保关闭 CARLA 环境，释放资源
        env.close()


# 程序入口：确保只有直接运行本脚本时才执行 main()
if __name__ == "__main__":
    main()
