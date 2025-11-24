# ==============================================
# 黑白棋（Reversi/Othello）强化学习训练代码
# 核心逻辑：黑棋采用随机策略，白棋采用自定义强化学习智能体（RL_QG_agent）
# 训练流程：环境注册→环境创建→智能体初始化→多局对战训练→结果统计→模型保存
# ==============================================

# 导入基础工具库
import random  # 用于黑棋的随机落子策略
import gym     # OpenAI Gym：强化学习环境标准框架，提供环境创建、交互接口
from gym.envs.registration import register  # Gym环境注册函数，用于自定义环境注册
import numpy as np  # 数值计算库，用于棋盘状态统计（如得分计算）

# 导入自定义模块
from gym.envs.reversi.reversi import ReversiEnv  # 自定义黑白棋环境类（实现棋盘规则、状态管理等）
from RL_QG_agent import RL_QG_agent  # 自定义强化学习智能体类（实现Q学习/其他RL算法）

# ==============================================
# 第一步：注册自定义黑白棋环境
# Gym要求自定义环境必须先注册，才能通过gym.make()创建实例
# ==============================================
register(
    id='Reversi8x8-v0',  # 环境唯一标识符（后续创建环境时使用）
    entry_point='gym.envs.reversi.reversi:ReversiEnv',  # 环境类的路径（包.模块:类名）
    kwargs={  # 传递给ReversiEnv类的初始化参数
        'player_color': 'black',  # 初始玩家颜色（黑棋先行）
        'opponent': 'random',     # 对手类型（此处为随机策略对手，即黑棋是随机玩家）
        'observation_type': 'numpy3c',  # 观测数据类型：3通道numpy数组（可能分别存储黑棋、白棋、空位置）
        'illegal_place_mode': 'lose',   # 非法落子处理方式：直接判负
        'board_size': 8  # 棋盘尺寸（8x8标准黑白棋）
    },
    max_episode_steps=1000,  # 每局最大步数限制（防止无限循环）
)

# 验证环境是否注册成功
envs = [spec.id for spec in gym.envs.registry.all()]  # 获取所有已注册的环境ID列表
print("Reversi8x8-v0 是否注册成功：", 'Reversi8x8-v0' in envs)  # 打印注册结果

# ==============================================
# 第二步：创建黑白棋环境实例
# 基于已注册的环境ID，创建可交互的环境对象
# ==============================================
env = gym.make(
    'Reversi8x8-v0',  # 目标环境ID（必须与注册时一致）
    player_color='black',  # 覆盖注册时的参数：初始玩家为黑棋
    opponent='random',     # 覆盖注册时的参数：对手为随机策略
    observation_type='numpy3c',  # 观测类型：3通道numpy数组
    illegal_place_mode='lose'    # 非法落子直接判负
)

# ==============================================
# 第三步：初始化强化学习智能体（白棋玩家）
# ==============================================
agent = RL_QG_agent()  # 实例化自定义RL智能体（控制白棋）
agent.init_model()     # 初始化智能体的模型（如Q表、神经网络等）
agent.load_model()     # 加载预训练模型（若存在，可基于历史模型继续训练）

# ==============================================
# 第四步：设置训练参数
# ==============================================
max_epochs = 100  # 训练总局数（可根据需求调整，如1000局、10000局）
render_interval = 10  # 渲染间隔：每1局渲染一次棋盘（便于可视化训练过程）

# ==============================================
# 第五步：训练主循环（核心逻辑）
# 外层循环：每一局对战；内层循环：每一步落子（黑棋→白棋交替）
# ==============================================
for i_episode in range(max_epochs):
    # 重置环境：开始新一局，返回初始观测（初始棋盘状态）
    observation = env.reset()  
    
    # 每局内部的步数循环（最大100步，防止超时）
    for t in range(100):  
        ################### 黑棋回合（随机策略玩家） ###################
        # 按渲染间隔判断是否渲染棋盘（此处每局都渲染）
        if i_episode % render_interval == 0:
            env.render()  # 可视化棋盘状态（显示当前落子位置、棋盘布局）
        
        enables = env.possible_actions  # 获取黑棋当前的所有合法落子位置（列表形式）
        
        # 黑棋选择动作：无合法动作则"pass"（跳过回合）
        if len(enables) == 0:
            # 动作编码：棋盘大小的平方+1 表示pass（8x8棋盘→64+1=65为pass动作）
            action_black = env.board_size**2 + 1  
        else:
            action_black = random.choice(enables)  # 有合法动作时，随机选择一个落子
        
        # 执行黑棋动作：将选择的动作传入环境，获取反馈
        # observation：执行动作后的新棋盘状态（观测）
        # reward：动作带来的即时奖励（此处未显式使用，可能在智能体训练中用到）
        # done：是否结束当前局（True=游戏结束，False=继续）
        # info：额外信息（如落子是否合法、当前玩家等，可选）
        observation, reward, done, info = env.step(action_black)
        
        if done:  # 若黑棋动作后游戏结束（如双方都pass或棋盘满），跳出步数循环
            break

        ################### 白棋回合（强化学习智能体） ###################
        # 渲染棋盘（与黑棋回合一致，每局都显示）
        if i_episode % render_interval == 0:
            env.render()
        
        enables = env.possible_actions  # 获取白棋当前的所有合法落子位置
        
        # 智能体选择动作：无合法动作则"pass"
        if not enables:  # 等价于 len(enables) == 0
            action_white = env.board_size ** 2 + 1  # pass动作编码
        else:
            # 智能体根据当前观测（棋盘状态）和合法动作，选择最优落子
            action_white = agent.place(observation, enables)
        
        # 执行白棋动作：获取环境反馈
        observation, reward, done, info = env.step(action_white)
        
        if done:  # 若白棋动作后游戏结束，跳出步数循环
            break

    # ==============================================
    # 每局结束后：统计结果并打印
    # ==============================================
    print(f"\n第 {i_episode+1} 局结束，总步数：{t+1}")  # t从0开始，需+1表示实际步数
    
    # 计算得分：棋盘上1代表黑棋，-1代表白棋，求和得到各自棋子数量
    black_score = np.sum(env.board == 1)    # 黑棋得分（棋子数）
    white_score = np.sum(env.board == -1)   # 白棋得分（棋子数）
    print(f"黑棋（随机策略）：{black_score} 子，白棋（RL智能体）：{white_score} 子")
    
    # 判断胜负结果
    if black_score > white_score:
        print("本局结果：黑棋获胜！")
    elif black_score < white_score:
        print("本局结果：白棋获胜！")
    else:
        print("本局结果：平局！")

# ==============================================
# 第六步：训练结束后处理
# ==============================================
agent.save_model()  # 保存训练后的智能体模型（覆盖原有模型或保存为新文件）
env.close()         # 关闭环境，释放资源（如渲染窗口、内存等）
print(f"\n训练完成！共进行 {max_epochs} 局对战")