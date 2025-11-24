
import random
import gym
from gym.envs.registration import register
import numpy as np
# 导入自定义环境和智能体
from gym.envs.reversi.reversi import ReversiEnv
from RL_QG_agent import RL_QG_agent

# 注册黑白棋环境
register(
    id='Reversi8x8-v0',
    entry_point='gym.envs.reversi.reversi:ReversiEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'random',
        'observation_type': 'numpy3c',
        'illegal_place_mode': 'lose',
        'board_size': 8
    },
    max_episode_steps=1000,  # 每局最大步数
)

# 验证环境注册
envs = [spec.id for spec in gym.envs.registry.all()]
print("Reversi8x8-v0 是否注册成功：", 'Reversi8x8-v0' in envs)

# 创建环境
env = gym.make(
    'Reversi8x8-v0',
    player_color='black',
    opponent='random',
    observation_type='numpy3c',
    illegal_place_mode='lose'
)

# 初始化智能体（白棋）
agent = RL_QG_agent()
agent.init_model()
agent.load_model()

# 训练参数
max_epochs = 10  # 训练局数（可修改）
render_interval = 1  # 每局都渲染

# 训练主循环
for i_episode in range(max_epochs):
    observation = env.reset()  # 重置环境
    for t in range(100):  # 每局最大步数
        ################### 黑棋回合（随机策略） ###################
        if i_episode % render_interval == 0:
            env.render()  # 渲染棋盘
        enables = env.possible_actions  # 合法动作

        # 选择黑棋动作
        if len(enables) == 0:
            action_black = env.board_size**2 + 1  # pass
        else:
            action_black = random.choice(enables)

        # 执行黑棋动作
        observation, reward, done, info = env.step(action_black)
        if done:
            break

        ################### 白棋回合（智能体策略） ###################
        if i_episode % render_interval == 0:
            env.render()
        enables = env.possible_actions  # 白棋合法动作

        # 智能体选择动作
        if not enables:
            action_white = env.board_size ** 2 + 1  # pass
        else:
            action_white = agent.place(observation, enables)

        # 执行白棋动作
        observation, reward, done, info = env.step(action_white)
        if done:
            break

    # 游戏结束，打印结果
    print(f"\n第 {i_episode+1} 局结束，步数：{t+1}")
    black_score = np.sum(env.board == 1)
    white_score = np.sum(env.board == -1)
    print(f"黑棋：{black_score} 子，白棋：{white_score} 子")
    if black_score > white_score:
        print("黑棋获胜！")
    elif black_score < white_score:
        print("白棋获胜！")
    else:
        print("平局！")

# 保存最终模型并关闭环境
agent.save_model()
env.close()
print(f"\n训练完成！共进行 {max_epochs} 局")

