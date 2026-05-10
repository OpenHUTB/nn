import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import os
from environment import RobotNavigationEnv
from dqn_agent import DQNAgent
from visualization import NavigationVisualizer
from config import Config

def train(episodes=None, visualize=None):
    config = Config()

    # 命令行参数覆盖配置
    if episodes is not None:
        config.EPISODES = episodes
    if visualize is not None:
        config.VISUALIZE = visualize

    # 创建结果目录
    os.makedirs(config.RESULT_DIR, exist_ok=True)

    env = RobotNavigationEnv()
    agent = DQNAgent(config.STATE_SIZE, config.ACTION_SIZE)
    visualizer = NavigationVisualizer()

    all_rewards = []
    all_distances = []
    all_lengths = []

    print("=" * 60)
    print("开始训练DQN导航代理...")
    print(f"训练轮数: {config.EPISODES}")
    print(f"最大步数: {config.MAX_STEPS}")
    print(f"状态维度: {config.STATE_SIZE}")
    print(f"动作数量: {config.ACTION_SIZE}")
    print(f"可视化: {'开启' if config.VISUALIZE else '关闭'}")
    print("=" * 60)

    for episode in range(config.EPISODES):
        state = env.reset()
        total_reward = 0
        distance_history = []
        
        visualizer.init_plot()
        visualizer.draw_obstacles(env.get_obstacles())
        visualizer.draw_target(env.get_target_pose()[0], env.get_target_pose()[1])
        visualizer.reset_path()
        
        for step in range(config.MAX_STEPS):
            action = agent.act(state)
            next_state, reward, done, distance = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            distance_history.append(distance)
            
            if episode % config.PLOT_INTERVAL == 0 and config.VISUALIZE:
                x, y, theta = env.get_robot_pose()
                lidar_data = next_state[:config.LIDAR_ANGLES] * config.LIDAR_RANGE
                
                visualizer.draw_robot(x, y, theta)
                visualizer.draw_path(x, y)
                visualizer.draw_lidar(x, y, theta, lidar_data)
            
            if done:
                break
        
        loss = agent.replay(config.BATCH_SIZE)
        
        all_rewards.append(total_reward)
        all_distances.append(distance_history[-1] if distance_history else config.MAP_WIDTH)
        all_lengths.append(step + 1)
        
        if episode % 10 == 0:
            avg_reward = np.mean(all_rewards[-10:])
            avg_distance = np.mean(all_distances[-10:])
            print(f"Episode {episode:4d} | Reward: {total_reward:6.2f} | Avg Reward: {avg_reward:6.2f} | "
                  f"Distance: {distance_history[-1]:.2f} | Avg Distance: {avg_distance:.2f} | "
                  f"Steps: {step+1:3d} | Epsilon: {agent.epsilon:.4f}")
        
        if episode % config.PLOT_INTERVAL == 0 and config.VISUALIZE:
            visualizer.save_figure(f'navigation_episode_{episode}.png')
            print(f"  -> 保存导航截图: navigation_episode_{episode}.png")
    
    print("=" * 60)
    print("训练完成！")
    print("=" * 60)
    
    # 保存模型
    agent.save_model()
    print(f"模型已保存: dqn_navigation_model.pth")
    
    # 绘制训练历史
    visualizer.plot_training_history(all_rewards, all_distances, all_lengths)
    print(f"训练历史图已保存: {config.RESULT_DIR}/training_history.png")
    
    print("=" * 60)
    print(f"所有训练结果已保存到 {config.RESULT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    train()