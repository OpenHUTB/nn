import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

print("正在导入模块...")

from config import Config
print("导入 config 成功")

from environment import RobotNavigationEnv
print("导入 environment 成功")

from dqn_agent import DQNAgent
print("导入 dqn_agent 成功")

from visualization import NavigationVisualizer
print("导入 visualization 成功")

print("开始训练...")

config = Config()
config.EPISODES = 50

os.makedirs(config.RESULT_DIR, exist_ok=True)

env = RobotNavigationEnv()
agent = DQNAgent(config.STATE_SIZE, config.ACTION_SIZE)
visualizer = NavigationVisualizer()

all_rewards = []

for episode in range(config.EPISODES):
    state = env.reset()
    total_reward = 0
    
    for step in range(config.MAX_STEPS):
        action = agent.act(state)
        next_state, reward, done, distance = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    agent.replay(config.BATCH_SIZE)
    all_rewards.append(total_reward)
    
    if episode % 5 == 0:
        avg_reward = np.mean(all_rewards[-5:])
        print("Episode %3d | Reward: %6.2f | Avg: %6.2f | Epsilon: %.4f" % 
              (episode, total_reward, avg_reward, agent.epsilon))

print("训练完成！")

fig, ax = plt.subplots()
ax.plot(all_rewards)
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.set_title('Training Reward History')
plt.savefig(os.path.join(config.RESULT_DIR, 'training_reward.png'))
print("训练历史图已保存到", os.path.join(config.RESULT_DIR, 'training_reward.png'))

agent.save_model()
print("模型已保存")