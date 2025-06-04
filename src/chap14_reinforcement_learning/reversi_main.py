import gym
import random
import numpy as np

from RL_QG_agent import RL_QG_agent  # 导入强化学习智能体

# 创建初始环境并重置
env = gym.make('Reversi8x8-v0')
env.reset()

# 初始化智能体并加载预训练模型
agent = RL_QG_agent()
agent.load_model()

# 设置最大训练轮数
max_epochs = 100

# 主训练循环（控制训练的总轮数）
for i_episode in range(max_epochs):
    # 初始化棋局，返回初始 observation（3x8x8 的状态表示）
    #3个通道分别表示：黑棋位置、白棋位置、当前玩家
    observation = env.reset()
    
    # 每局最多进行 100 步操作（黑白双方交替下棋）
    for t in range(100):
        action = [1, 2]  # 初始化 action，占位，稍后会被赋真实值
        # action[0]: 表示棋盘上的位置（0~63），或特殊值表示“跳过”
        # action[1]: 表示棋子的颜色，0 表示黑棋，1 表示白棋

        ################### 黑棋（随机策略） ###################
        env.render()  # 打印当前棋盘状态
        
        # 获取当前黑棋可落子的位置列表
        # enables是一个包含合法位置索引的列表
        enables = env.possible_actions  
        
        # 判断是否有合法落子的位置
        if len(enables) == 0:
            # 没有可落子位置，则选择“跳过”操作，编码为 board_size^2 + 1
            action_ = env.board_size**2 + 1
        else:
            # 随机选择一个合法位置
            action_ = random.choice(enables)
        action[0] = action_
        action[1] = 0  # 设置为黑棋
        # 黑棋落子并更新环境状态，返回新状态、奖励、是否结束等信息
        observation, reward, done, info = env.step(action)

        ################### 白棋（智能体策略） ###################
        env.render()  # 打印当前棋盘状态
        enables = env.possible_actions  # 获取当前白棋可落子的位置列表
        if len(enables) == 0:
            # 无法落子，执行“跳过”操作
            action_ = env.board_size ** 2 + 1
        else:
            # 使用训练好的智能体模型选择最佳落子位置
            action_ = agent.place(observation, enables)
        action[0] = action_
        action[1] = 1  # 设置为白棋
        # 白棋落子并更新环境状态
        observation, reward, done, info = env.step(action)

        # 如果对局结束
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            # 统计黑棋得分（棋盘中为1的个数）
            black_score = len(np.where(env.state[0,:,:]==1)[0])
            if black_score > 32:
                print("黑棋赢了！")
            else:
                print("白棋赢了！")
            print(black_score)  # 打印黑棋得分
            break #结束代码
