#!/usr/bin/env python3
"""
修复中文显示的增强版DQN训练脚本
"""

import os
import sys
import time

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from client.drone_client import DroneClient
from agents.enhanced_dqn_agent import EnhancedDQNAgent
from agents.dqn.config import DQNConfig


def main():
    print("🚁 修复版增强DQN训练脚本")
    print("="*50)

    # 先修复中文显示
    print("\n1. 检查系统环境...")
    import platform
    print(f"操作系统: {platform.system()}")

    # 创建配置
    config = DQNConfig()
    config.MAX_EPISODES = 100  # 减少测试轮数
    config.HIDDEN_DIM = 256
    config.LEARNING_RATE = 1e-3
    config.BUFFER_SIZE = 20000

    # 创建客户端
    print("\n2. 创建无人机客户端...")
    client = DroneClient(interval=10, root_path='./')

    # 创建修复版智能体
    print("3. 创建修复版DQN智能体...")
    agent = EnhancedDQNAgent(
        client,
        move_type='velocity',
        config=config,
        enable_visualization=True  # 启用可视化
    )

    # 训练智能体
    print("\n4. 开始训练...")
    print("已启用的功能:")
    print("  ✓ 增强奖励函数（多组件）")
    print("  ✓ 训练完成后显示可视化")
    print("  ✓ 修复中文显示问题")
    print("  ✓ 优先经验回放")
    print("  ✓ 学习率调度")
    print("  ✓ 梯度裁剪")
    print("  ✓ Huber损失（鲁棒训练）")
    print("  ✓ 最佳模型检查点")
    print("\n训练中，请稍候...")

    # 开始训练
    start_time = time.time()
    agent.train(episodes=config.MAX_EPISODES, save_path='./models')
    training_time = time.time() - start_time

    print(f"\n训练完成，用时 {training_time:.2f} 秒")

    # 训练完成后显示可视化
    print("\n5. 生成训练可视化图表...")
    if config.VISUALIZATION['enable']:
        # 重新绘制最终图表
        print("正在生成可视化图表...")
        agent.env.update_visualization(len(agent.episode_rewards), agent.episode_rewards)

    # 测试训练好的模型
    print("\n6. 测试训练好的模型...")
    agent.run(episodes=5)

    # 关闭环境
    print("\n7. 关闭环境...")
    agent.close()
    client.destroy()

    print("\n训练完成！")
    print("请查看以下内容：")
    print("  - ./models/: 保存的模型文件")
    print("  - ./training_plots/: 训练可视化图表")
    print("  - 控制台输出：训练统计和结果")


if __name__ == "__main__":
    main()