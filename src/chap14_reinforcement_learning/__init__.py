from gym.envs.registration import registry, register, make, spec# 导入Gym环境注册相关模块

# Algorithmic Environments - 算法类环境
# 设计用于测试序列学习、记忆和模式识别能力

# 复制任务：智能体需记忆并复制输入序列
register(
    id = 'Copy-v0',
    entry_point = 'gym.envs.algorithmic:CopyEnv',# 环境实现类
    max_episode_steps = 200,# 单次训练最大步数
    reward_threshold = 25.0,# 视为任务成功的奖励阈值
)
#重复复制任务：需记忆并多次复制输入序列
register(
    id = 'RepeatCopy-v0', # 重复复制任务
    entry_point = 'gym.envs.algorithmic:RepeatCopyEnv',
    max_episode_steps = 200,# 设置单轮训练的最大时间步数限制
    reward_threshold = 75.0,# 更高阈值反映任务复杂性增加
)

# 反向加法任务：执行反向数字加法运算
register(
    id = 'ReversedAddition-v0',
    entry_point = 'gym.envs.algorithmic:ReversedAdditionEnv',
    kwargs = {'rows' : 2},# 自定义参数：操作数行数
    max_episode_steps = 200,#限制最大步数，防止陷入循环
    reward_threshold = 25.0,
)
# 三操作数反向加法任务
register(
    id = 'ReversedAddition3-v0',
    entry_point = 'gym.envs.algorithmic:ReversedAdditionEnv',
    kwargs = {'rows' : 3},# 三行操作数增加难度
    max_episode_steps = 200,
    reward_threshold = 25.0,
)
# 重复输入检测任务：识别重复出现的输入元素
register(
    id = 'DuplicatedInput-v0',  # 环境唯一标识符
    entry_point = 'gym.envs.algorithmic:DuplicatedInputEnv',  # 环境类的导入路径
    max_episode_steps = 200,
    reward_threshold = 9.0,# 相对较低的阈值
)
# 序列反转任务：将输入序列完全反转输出
register(
    id = 'Reverse-v0',
    entry_point = 'gym.envs.algorithmic:ReverseEnv',
    max_episode_steps = 200,
    reward_threshold = 25.0,
)

# Classic Control Environments - 经典控制环境
# 基于物理模型的简单控制问题，常用于算法基准测试
# 经典控制环境：基于数学模型的控制问题

# 基础版倒立摆平衡任务
register(
    id = 'CartPole-v0',  
    entry_point = 'gym.envs.classic_control:CartPoleEnv',
    max_episode_steps = 200,  # 平衡200步视为成功
    reward_threshold = 195.0,# 接近最大理论值200
)

# 高难度倒立摆版本（延长测试时长）
register(
    id = 'CartPole-v1',  # 更高难度版本
    entry_point = 'gym.envs.classic_control:CartPoleEnv',
    max_episode_steps = 500,  # 500步达标
    reward_threshold = 475.0,
)
# 山车任务：利用动量爬坡
register(
    id='MountainCar-v0',  # 山车任务：爬坡
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=200,
    reward_threshold=-110.0,  # 负数表示尽量减少步数
)

# 山车连续控制版本（精细油门控制）
register(
    id='MountainCarContinuous-v0',    # 连续动作版本
    entry_point='gym.envs.classic_control:Continuous_MountainCarEnv',
    max_episode_steps=999,
    reward_threshold=90.0,
)

register(
    id='Pendulum-v0',   # 钟摆任务：摆到垂直位置
    entry_point='gym.envs.classic_control:PendulumEnv',
    max_episode_steps = 200,
)

register(
    id='Acrobot-v1',     # 双连杆机械臂任务
    entry_point='gym.envs.classic_control:AcrobotEnv',
    max_episode_steps=500,
)

# Box2d物理引擎环境：用于复杂物理模拟任务
# ----------------------------------------
# 该模块使用Box2D物理引擎实现高保真度的物理模拟
# 特别适合连续控制任务和刚体动力学仿真

# 注册名为'LunarLander-v2'的自定义环境
register(
    id='LunarLander-v2',     # 环境唯一标识符，格式：<任务名>-<版本号>
                            # 此处是月球着陆器环境第2版
    
    # 指定环境类的入口点路径：
    # gym.envs.box2d模块下的LunarLander类
    entry_point='gym.envs.box2d:LunarLander',  
    
    # 每个episode的最大步数限制（防止无限运行）
    max_episode_steps=1000,  
    
    # 奖励阈值：当episode累计奖励超过200时
    # 视为任务成功（着陆器安全着陆）
    reward_threshold=200,    
)
# 月球着陆器连续动作版本
register(
    id='LunarLanderContinuous-v2',  
    entry_point='gym.envs.box2d:LunarLanderContinuous',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='BipedalWalker-v2',   # 双足步行机器人
    entry_point='gym.envs.box2d:BipedalWalker',
    max_episode_steps=1600,# 长时程任务
    reward_threshold=300,     # 完成步行得分
)

register(
    id='BipedalWalkerHardcore-v2',   # 双足步行困难版（复杂地形）
    entry_point='gym.envs.box2d:BipedalWalkerHardcore',
    max_episode_steps=2000,# 更长的步数限制
    reward_threshold=300,  
)

register(
    id='CarRacing-v0',    # 赛车游戏
    entry_point='gym.envs.box2d:CarRacing',
    max_episode_steps=1000,
    reward_threshold=900,   # 完成赛道得分
)

# Toy Text
# 文本类简单环境：离散状态空间
# ----------------------------------------

register(
    id='Blackjack-v0',    # 21点游戏
    entry_point='gym.envs.toy_text:BlackjackEnv',
)

register(
    id='KellyCoinflip-v0',   # 凯利判赌任务
    entry_point='gym.envs.toy_text:KellyCoinflipEnv',
    reward_threshold=246.61,
)
register(
    id='KellyCoinflipGeneralized-v0',    # 通用凯利判赌
    entry_point='gym.envs.toy_text:KellyCoinflipGeneralizedEnv',
)

register(
    id='FrozenLake-v0',   # 冰湖行走任务
    entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
    kwargs = {'map_name' : '4x4'},   # 4x4网格
    max_episode_steps = 100,
    reward_threshold = 0.78, # optimum = .8196   # 成功到达目标的平均奖励
)

# 冰湖行走（8x8网格版）
register(
    id='FrozenLake8x8-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '8x8'},
    max_episode_steps=200,
    reward_threshold=0.99, # optimum = 1
)
# 悬崖行走路径规划任务
register(
    id='CliffWalking-v0',
    entry_point='gym.envs.toy_text:CliffWalkingEnv',
)
# N链问题（探索-利用权衡）
register(
    id='NChain-v0',
    entry_point='gym.envs.toy_text:NChainEnv',
    max_episode_steps=1000,
)

# 注册后可通过gym.make('Roulette-v0')创建环境实例
# 轮盘赌模拟环境
register(
    id='Roulette-v0',  # 环境唯一标识符（在代码中引用的名称
    entry_point='gym.envs.toy_text:RouletteEnv',  # 环境类的导入路径
                                                  # 格式为：'包名:类名'

    max_episode_steps=100,              # 每个episode的最大步数
                                        # 达到此步数后，环境自动终止
                                        # 防止无限循环，控制训练复杂度
)
# 出租车调度任务（乘客接送）
register(
    id='Taxi-v2',  # 环境的唯一标识符
    entry_point='gym.envs.toy_text.taxi:TaxiEnv',  # 指定环境类的位置
    reward_threshold=8, # optimum = 8.46
    max_episode_steps=200,  # 设置每个episode的最大步数为200
)
# 数字猜测游戏
register(
    id='GuessingGame-v0',
    entry_point='gym.envs.toy_text.guessing_game:GuessingGame',
    max_episode_steps=200,
)
# 热冷游戏：基于反馈的目标搜索
register(
    id='HotterColder-v0',
    entry_point='gym.envs.toy_text.hotter_colder:HotterColder',
    max_episode_steps=200,
)

# MuJoCo Environments - 机器人控制环境
# 基于MuJoCo物理引擎的连续控制任务（需要独立许可证）

# 2D机械臂到达目标位置

register(
    id='Reacher-v1',
    entry_point='gym.envs.mujoco:ReacherEnv',
    max_episode_steps=50,# 较短步数限制
    reward_threshold=-3.75,# 负阈值表示最小化距离
)
# 物体推动任务
register(
    id='Pusher-v0',
    entry_point='gym.envs.mujoco:PusherEnv',
    max_episode_steps=100,
    reward_threshold=0.0,# 尚未定义明确阈值
)
# 物体投掷任务
register(
    id='Thrower-v0',
    entry_point='gym.envs.mujoco:ThrowerEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)
# 物体击打任务
register(
    id='Striker-v0',
    entry_point='gym.envs.mujoco:StrikerEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)
# 单级倒立摆
register(
    id='InvertedPendulum-v1',
    entry_point='gym.envs.mujoco:InvertedPendulumEnv',
    max_episode_steps=1000,
    reward_threshold=950.0,# 高精度控制要求
)
# 二级倒立摆（更复杂）
register(
    id='InvertedDoublePendulum-v1',
    entry_point='gym.envs.mujoco:InvertedDoublePendulumEnv',
    max_episode_steps=1000,
    reward_threshold=9100.0,# 显著提高的阈值
)
# 猎豹奔跑任务（速度最大化）
register(
    id='HalfCheetah-v1',
    entry_point='gym.envs.mujoco:HalfCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,# 高速要求
)
# 单腿跳跃机器人
register(
    id='Hopper-v1',
    entry_point='gym.envs.mujoco:HopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,# 持续跳跃要求
)
# 游泳者运动控制
register(
    id='Swimmer-v1',
    entry_point='gym.envs.mujoco:SwimmerEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,# 相对较低阈值
)
# 双腿行走机器人
register(
    id = 'Walker2d-v1',
    max_episode_steps = 1000,
    entry_point = 'gym.envs.mujoco:Walker2dEnv',
)
# 蚂蚁机器人运动控制（四足）
register(
    id = 'Ant-v1',
    entry_point = 'gym.envs.mujoco:AntEnv',
    max_episode_steps = 1000,
    reward_threshold = 6000.0,
)

register(
    id='Humanoid-v1',
    entry_point='gym.envs.mujoco:HumanoidEnv',
    max_episode_steps=1000,
)
# 类人机器人站立任务
register(
    id='HumanoidStandup-v1',
    entry_point='gym.envs.mujoco:HumanoidStandupEnv',
    max_episode_steps=1000,
)

# Atari Environments - 雅达利游戏环境
# 通过Arcade Learning Environment封装的2600款经典游戏

# 遍历所有支持的雅达利游戏
# # print ', '.join(["'{}'".format(name.split('.')[0]) for name in atari_py.list_games()])
for game in ['air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
    'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
    'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk',
    'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
    'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
    'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
    'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
    'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
    'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']:
    for obs_type in ['image', 'ram']:
        # 生成标准环境名称（如 SpaceInvaders-v0）
        name = ''.join(g.capitalize() for g in game.split('_'))
        if obs_type == 'ram':
            name = f'{name}-ram'  # RAM观测版本添加-ram后缀
# 处理特殊游戏的非确定性
        nondeterministic = False
        if game == 'elevator_action' and obs_type == 'ram':
            # ElevatorAction-ram-v0 seems to yield slightly
            # non-deterministic observations about 10% of the time. We
            # should track this down eventually, but for now we just
            # mark it as nondeterministic.
            nondeterministic = True
# 注册基础版本 (v0)
        register(
            id='{}-v0'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'repeat_action_probability': 0.25},# 动作重复概率（模拟硬件延迟）
            max_episode_steps=10000,
            nondeterministic=nondeterministic,
        )
# 注册改进版本 (v4)
        register(
            id='{}-v4'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type},# 移除动作重复概率
            max_episode_steps=100000,# 更长步数限制
            nondeterministic=nondeterministic,
        )
# 设置游戏特定帧跳过策略
        # Standard Deterministic (as in the original DeepMind paper)
        if game == 'space_invaders':
            frameskip = 3
        else:
            frameskip = 4
# 确定性帧跳过版本 (v0)
        # Use a deterministic frame skip.
        register(
            id='{}Deterministic-v0'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': frameskip, 'repeat_action_probability': 0.25},# 固定帧跳过
            max_episode_steps=100000,
            nondeterministic=nondeterministic,
        )
# 确定性帧跳过版本 (v4)
        register(
            id='{}Deterministic-v4'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': frameskip},
            max_episode_steps=100000,
            nondeterministic=nondeterministic,
        )
# 无帧跳过版本 (v0)
        register(
            id='{}NoFrameskip-v0'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1, 'repeat_action_probability': 0.25}, # A frameskip of 1 means we get every frame
            max_episode_steps=frameskip * 100000,# 每帧都处理，按比例增加步数限制
            nondeterministic=nondeterministic,
        )
# 无帧跳过版本 (v4)
        # No frameskip. (Atari has no entropy source, so these are
        # deterministic environments.)
        register(
            id='{}NoFrameskip-v4'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1}, # A frameskip of 1 means we get every frame
            max_episode_steps=frameskip * 100000,
            nondeterministic=nondeterministic,
        )

# Board games
# 棋类游戏环境

#9x9围棋环境
register(
    id='Go9x9-v0',
    entry_point='gym.envs.board_game:GoEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'pachi:uct:_2400',# 使用Pachi AI作为对手
        'observation_type': 'image3c',# 图像观测
        'illegal_move_mode': 'lose',# 非法移动直接判负
        'board_size': 9,
    },
    # The pachi player seems not to be determistic given a fixed seed.
    # (Reproduce by running 'import gym; h = gym.make('Go9x9-v0'); h.seed(1); h.reset(); h.step(15); h.step(16); h.step(17)' a few times.)
    #
    # This is probably due to a computation time limit.
    nondeterministic=True,# 因AI计算时间限制导致非确定性
)
#19x19标准围棋环境
register(
    id='Go19x19-v0',
    entry_point='gym.envs.board_game:GoEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'pachi:uct:_2400',
        'observation_type': 'image3c',
        'illegal_move_mode': 'lose',
        'board_size': 19,# 标准棋盘尺寸
    },
    nondeterministic=True,
)
# 9x9六边形棋环境
register(
    id='Hex9x9-v0',
    entry_point='gym.envs.board_game:HexEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'random',
        'observation_type': 'numpy3c',
        'illegal_move_mode': 'lose',
        'board_size': 9,
    },
)

# Debugging
# ----------------------------------------

register(
    id='OneRoundDeterministicReward-v0',
    entry_point='gym.envs.debugging:OneRoundDeterministicRewardEnv',
    local_only=True
)

register(
    id='TwoRoundDeterministicReward-v0',
    entry_point='gym.envs.debugging:TwoRoundDeterministicRewardEnv',
    local_only=True
)

register(
    id='OneRoundNondeterministicReward-v0',
    entry_point='gym.envs.debugging:OneRoundNondeterministicRewardEnv',
    local_only=True
)

register(
    id='TwoRoundNondeterministicReward-v0',
    entry_point='gym.envs.debugging:TwoRoundNondeterministicRewardEnv',
    local_only=True,
)

# Parameter tuning
# ----------------------------------------
register(
    id='ConvergenceControl-v0',
    entry_point='gym.envs.parameter_tuning:ConvergenceControl',
)

register(
    id='CNNClassifierTraining-v0',
    entry_point='gym.envs.parameter_tuning:CNNClassifierTraining',
)

# Safety
# ----------------------------------------

# interpretability envs
register(
    id = 'PredictActionsCartpole-v0',
    entry_point = 'gym.envs.safety:PredictActionsCartpoleEnv',
    max_episode_steps=200,
)

register(
    id = 'PredictObsCartpole-v0',
    entry_point = 'gym.envs.safety:PredictObsCartpoleEnv',
    max_episode_steps=200,
)

# semi_supervised envs
    # probably the easiest:
register(
    id='SemisuperPendulumNoise-v0',                # 环境标识符，用于gym.make()
    entry_point='gym.envs.safety:SemisuperPendulumNoiseEnv',  # 环境类路径
    max_episode_steps=200,                 # 每个episode最大步数
                                               # 与标准CartPole保持一致
)
    # somewhat harder because of higher variance:
register(
    id='SemisuperPendulumRandom-v0',
    entry_point='gym.envs.safety:SemisuperPendulumRandomEnv',
    max_episode_steps=200,
)
    # probably the hardest because you only get a constant number of rewards in total:
register(
    id='SemisuperPendulumDecay-v0',
    entry_point='gym.envs.safety:SemisuperPendulumDecayEnv',
    max_episode_steps=200,
)

# off_switch envs
register(
    id='OffSwitchCartpole-v0',
    entry_point='gym.envs.safety:OffSwitchCartpoleEnv',
    max_episode_steps = 200,
)

# 注册安全环境：带关闭开关的倒立摆概率环境
register(
    id='OffSwitchCartpoleProb-v0',
    entry_point = 'gym.envs.safety:OffSwitchCartpoleProbEnv',
    max_episode_steps = 200,
)

# 注册黑白棋（奥赛罗）游戏环境
register(
    id='Reversi8x8-v0',   # 环境唯一标识符，指定8x8棋盘版本
    entry_point = 'gym.envs.reversi:ReversiEnv',  # 入口点指向黑白棋模块
    kwargs = {  
        'player_color': 'black',
        'opponent': 'random',
        'observation_type': 'numpy3c',
        'illegal_place_mode': 'lose',
        'board_size': 8,
    }
)
