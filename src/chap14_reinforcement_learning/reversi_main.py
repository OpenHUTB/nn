# 导入OpenAI Gym库，提供标准化的强化学习环境接口
import gym
# 导入随机数生成库，用于实现随机策略
import random
# 导入NumPy库，用于高效的数值计算和数组操作
import numpy as np
# 导入自定义的强化学习智能体类
from RL_QG_agent import RL_QG_agent 

# 创建黑白棋环境实例（8x8标准棋盘）
env = gym.make('Reversi8x8-v0')  # 使用Gym接口创建特定环境
env.reset()  # 初始化环境状态

# 初始化强化学习智能体并加载预训练模型
agent = RL_QG_agent()  # 创建智能体实例，实现特定的学习算法
agent.load_model()  # 加载已训练的模型参数，加速学习过程

# 设置训练参数
max_epochs = 100  # 总共进行的训练局数，每局是完整的游戏

# 训练主循环
for i_episode in range(max_epochs):
    # 重置环境，开始新的一局游戏
    # observation: 3x8x8的张量，包含游戏状态信息
    # 3个通道分别表示: 黑棋位置、白棋位置、当前玩家
    observation = env.reset()
    
    # 单局游戏循环（最多100步，防止无限循环）
    for t in range(100):
        # 初始化动作，稍后会被具体动作覆盖
        action = [1, 2]  # action[0]: 落子位置(0-63)或特殊操作, action[1]: 棋子颜色(0=黑, 1=白)

        ################### 黑棋回合（使用随机策略） ###################
        env.render()  # 可视化当前棋盘状态，便于观察训练过程
        enables = env.possible_actions  # 获取当前合法的落子位置列表

        # 处理无合法动作的情况
        if len(enables) == 0:
            # 无合法落子位置，执行"跳过"操作
            action_ = env.board_size**2 + 1  # 特殊编码表示"跳过"
        else:
            # 随机策略：从合法动作中随机选择一个
            action_ = random.choice(enables)
        
        # 构建完整动作 [位置, 颜色]
        action[0] = action_  # 设置落子位置
        action[1] = 0  # 设置棋子颜色为黑色
        
        # 执行黑棋动作，更新环境状态
        # observation: 更新后的环境观测
        # reward: 执行动作后获得的即时奖励（通常为0，除非游戏结束）
        # done: 游戏是否结束
        # info: 包含额外信息的字典（如获胜方）
        observation, reward, done, info = env.step(action)

        ################### 白棋回合（使用智能体策略） ###################
        env.render()  # 再次可视化棋盘状态
        enables = env.possible_actions  # 获取白棋合法落子位置

        # 处理无合法动作的情况
        if not enables:
            action_ = env.board_size ** 2 + 1  # 执行"跳过"操作
        else:  
            # 使用训练好的智能体选择最佳动作
            # observation: 当前环境观测
            # enables: 合法动作列表
            # 返回: 选择的动作索引
            action_ = agent.place(observation, enables)
        
        # 构建完整动作
        action[0] = action_  # 设置落子位置
        action[1] = 1  # 设置棋子颜色为白色
        
        # 执行白棋动作，更新环境状态
        observation, reward, done, info = env.step(action)

        # 检查游戏是否结束
        if done:
            # 打印游戏结果摘要
            print(f"第 {i_episode+1} 局游戏在 {t+1} 步后结束")
            
            # 计算双方得分
            black_score = len(np.where(env.state[0, :, :] == 1)[0])  # 统计黑棋数量
            total_tiles = env.board_size ** 2  # 棋盘总格子数 (8x8=64)
            
            # 判断胜负
            if black_score > total_tiles / 2:  # 黑棋数量超过一半
                print("黑棋获胜！")
            elif black_score < total_tiles / 2:  # 白棋数量超过一半
                print("白棋获胜！")
            else:  # 双方棋子数量相等
                print("平局！")
            
            # 打印详细比分
            white_score = total_tiles - black_score
            print(f"比分: 黑棋 {black_score} - 白棋 {white_score}")
            
            break  # 结束当前游戏，开始下一局

# 清理环境资源
env.close()
print(f"训练完成！共进行了 {max_epochs} 局游戏")
