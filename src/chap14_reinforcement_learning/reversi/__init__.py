from gym.envs.reversi.reversi import ReversiEnv # 从OpenAI Gym的Reversi（黑白棋）环境实现中导入核心环境类
from gym.envs.registration import register  # 关键导入语句
# 注册黑白棋环境（8x8棋盘）
register(
    id='Reversi8x8-v0',
    entry_point='gym.envs.reversi:ReversiEnv',  # 确保reversi.py中存在ReversiEnv类
    kwargs={'board_size': 8},
    max_episode_steps=1000,
)