import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
from environment import RobotNavigationEnv
from dqn_agent import DQNAgent
from visualization import NavigationVisualizer
from config import Config

def test(visualize=None):
    config = Config()
    env = RobotNavigationEnv()
    agent = DQNAgent(config.STATE_SIZE, config.ACTION_SIZE)
    
    try:
        agent.load_model()
        print("模型加载成功！")
    except Exception as e:
        print("未找到预训练模型，使用随机策略...")
        agent.epsilon = 0.0
    
    visualizer = NavigationVisualizer()
    visualizer.init_plot()
    visualizer.draw_obstacles(env.get_obstacles())
    visualizer.draw_target(env.get_target_pose()[0], env.get_target_pose()[1])
    visualizer.reset_path()
    
    state = env.reset()
    total_reward = 0
    path_history = []
    
    for step in range(config.MAX_STEPS):
        action = agent.act(state)
        next_state, reward, done, distance = env.step(action)
        
        state = next_state
        total_reward += reward
        
        x, y, theta = env.get_robot_pose()
        path_history.append((x, y))
        
        lidar_data = next_state[:config.LIDAR_ANGLES] * config.LIDAR_RANGE
        
        visualizer.draw_robot(x, y, theta)
        visualizer.draw_path(x, y)
        visualizer.draw_lidar(x, y, theta, lidar_data)
        
        if done:
            break
    
    print("测试完成！")
    print("总奖励: %.2f" % total_reward)
    print("最终距离目标: %.2f" % distance)
    print("步数: %d" % (step + 1))
    
    visualizer.save_figure('test_navigation_result.png')
    visualizer.plot_lidar_heatmap(lidar_data)
    
    print("结果已保存到", config.RESULT_DIR)

if __name__ == "__main__":
    test()