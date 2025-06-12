from gym.envs.registration import registry, register, make, spec  # 导入Gym环境注册核心模块，用于创建和管理强化学习环境

### 一、算法类环境（Algorithmic Environments）
# 设计用于测试序列处理、记忆能力和模式识别算法，适合研究长短期记忆机制
# 特点：输入输出均为序列数据，重点考察时间依赖关系处理能力

# 1. 复制任务：智能体需记忆任意输入序列并完整复现
register(
    id='Copy-v0',             # 环境唯一标识，通过gym.make('Copy-v0')调用
    entry_point='gym.envs.algorithmic:CopyEnv',  # 环境类的路径（模块:类名）
    max_episode_steps=200,    # 单个episode的最大步数限制，防止无限循环
    reward_threshold=25.0,    # 平均奖励达到此值视为任务成功
)

# 2. 重复复制任务：记忆序列并按指定次数重复输出（难度高于基础复制）
register(
    id='RepeatCopy-v0',       # 环境唯一标识符，遵循Gym的命名约定（任务名+版本号）
    entry_point='gym.envs.algorithmic:RepeatCopyEnv',
    max_episode_steps=200,    # 与基础复制任务相同步数限制
    reward_threshold=75.0,    # 更高奖励阈值，反映任务复杂度提升
)

# 3. 反向加法任务：对逆序输入的数字执行加法运算（如输入[3,1,2]和[5,9]表示213+95）
register(
    id='ReversedAddition-v0', 
    entry_point='gym.envs.algorithmic:ReversedAdditionEnv',
    kwargs={'rows': 2},       # 输入参数：2行操作数（两数相加）
    max_episode_steps=200, # 每个episode的最大步数限制，超过则终止
    reward_threshold=25.0, # 判定任务成功的奖励阈值，达到则认为训练完成
)

# 三操作数反向加法（难度提升：处理三个数相加）
register(
    id='ReversedAddition3-v0',
    entry_point='gym.envs.algorithmic:ReversedAdditionEnv',
    kwargs={'rows': 3},       # 3行操作数，需同时处理三个数的加法进位
    max_episode_steps=200,    # 每个episode的最大步数限制
    reward_threshold=25.0,    # 认为任务解决的成功阈值
)

# 4. 重复输入检测：识别序列中重复出现的元素（考察模式识别能力）
register(
    id='DuplicatedInput-v0', # 环境的唯一标识符
    entry_point='gym.envs.algorithmic:DuplicatedInputEnv', # 指定环境类的导入路径
    max_episode_steps=200, # 设置该环境的最大步数限制
    reward_threshold=9.0,     # 较低阈值，因任务本质为二分类问题
)

# 5. 序列反转任务：将输入序列完全逆序输出（基础序列处理任务）
register(
    id='Reverse-v0',
    entry_point='gym.envs.algorithmic:ReverseEnv',
    max_episode_steps=200,
    reward_threshold=25.0,
)


### 二、经典控制环境（Classic Control Environments）
# 基于物理模型的数学控制问题，状态空间和动作空间维度较低，适合算法基准测试

# 1. CartPole 倒立摆平衡任务
register(
    id='CartPole-v0',         # 基础版本
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=200,    # 平衡200步视为成功
    reward_threshold=195.0,   # 成功阈值（接近最大可能奖励200）
)

register(
    id='CartPole-v1',         # 高难度版本（延长测试时间）
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=500,    # 需平衡500步
    reward_threshold=475.0,
)

# 2. 山车任务：利用动量爬坡（状态空间连续，动作空间离散）
register(
    id='MountainCar-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=200,
    reward_threshold=-110.0,  # 负数奖励表示需要减少到达目标的步数
)

# 山车连续控制版本（动作空间为连续值，控制油门力度）
register(
    id='MountainCarContinuous-v0',
    entry_point='gym.envs.classic_control:Continuous_MountainCarEnv',
    max_episode_steps=999,   # 每个episode的最大步数限制
    reward_threshold=90.0,   # 成功阈值
)

# 3. 钟摆任务：将摆锤从下垂位置摆到垂直向上位置
register(
    id='Pendulum-v0',
    entry_point='gym.envs.classic_control:PendulumEnv',
    max_episode_steps=200,
)

# 4. 双连杆机械臂：通过关节控制使末端到达目标位置（无奖励阈值，仅考察控制）
register(
    id='Acrobot-v1',
    entry_point='gym.envs.classic_control:AcrobotEnv',
    max_episode_steps=500,
)


### 三、Box2D物理引擎环境
# 使用Box2D库实现的高保真物理模拟，适合连续控制和刚体动力学研究

# 1. 月球着陆器：控制着陆器在月球表面安全着陆（离散动作版本）
register(
    id='LunarLander-v2',
    entry_point='gym.envs.box2d:LunarLander',
    max_episode_steps=1000,   # 长时程任务
    reward_threshold=200,     # 安全着陆的奖励阈值
)

# 月球着陆器（连续动作版本，控制推力大小）
register(
    id='LunarLanderContinuous-v2',
    entry_point='gym.envs.box2d:LunarLanderContinuous',
    max_episode_steps=1000,
    reward_threshold=200,
)

# 2. 双足步行机器人（普通地形）
register(
    id='BipedalWalker-v2',
    entry_point='gym.envs.box2d:BipedalWalker',
    max_episode_steps=1600,   # 超长步数限制
    reward_threshold=300,     # 完成步行路径的得分
)

# 双足步行机器人（困难版，含障碍物和复杂地形）
register(
    id='BipedalWalkerHardcore-v2',
    entry_point='gym.envs.box2d:BipedalWalkerHardcore',
    max_episode_steps=2000,   # 进一步延长步数
    reward_threshold=300,
)

# 3. 赛车游戏：控制赛车在赛道上行驶并获得高分
register(
    id='CarRacing-v0',
    entry_point='gym.envs.box2d:CarRacing',
    max_episode_steps=1000,
    reward_threshold=900,     # 完成赛道的高分要求
)


### 四、文本类玩具环境（Toy Text）
# 离散状态空间的简单环境，适合教学和算法初步验证

# 1. 21点游戏：经典扑克牌游戏（考察决策策略）
register(
    id='Blackjack-v0',
    entry_point='gym.envs.toy_text:BlackjackEnv',
)

# 2. 凯利判赌任务：基于概率的赌博决策（考察期望收益计算）
register(
    id='KellyCoinflip-v0',
    entry_point='gym.envs.toy_text:KellyCoinflipEnv',
    reward_threshold=246.61,  # 理论最优收益阈值
)

# 通用凯利判赌（可调整概率参数的扩展版本）
register(
    id='KellyCoinflipGeneralized-v0',
    entry_point='gym.envs.toy_text:KellyCoinflipGeneralizedEnv',
)

# 3. 冰湖行走：在结冰的湖面行走，避免掉入冰窟（4x4网格）
register(
    id='FrozenLake-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4'},  # 小尺寸网格，适合入门
    max_episode_steps=100,
    reward_threshold=0.78,      # 成功到达终点的平均奖励
)

# 冰湖行走（8x8网格版，难度显著提升）
register(
    id='FrozenLake8x8-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '8x8'},  # 标准围棋棋盘大小
    max_episode_steps=200,
    reward_threshold=0.99,      # 接近理论最优解
)

# 4. 悬崖行走：在悬崖边缘寻找最短路径（考察风险规避策略）
register(
    id='CliffWalking-v0',
    entry_point='gym.envs.toy_text:CliffWalkingEnv',
)

# 5. N链问题：长链状态空间中的探索-利用权衡（经典强化学习难题）
register(
    id='NChain-v0',
    entry_point='gym.envs.toy_text:NChainEnv',
    max_episode_steps=1000,
)

# 6. 轮盘赌模拟：概率驱动的赌博环境（考察随机策略）
register(
    id='Roulette-v0',
    entry_point='gym.envs.toy_text:RouletteEnv',
    max_episode_steps=100,      # 短周期任务
)

# 7. 出租车调度：在城市网格中接送乘客（状态空间复杂的经典问题）
register(
    id='Taxi-v2',
    entry_point='gym.envs.toy_text.taxi:TaxiEnv',
    reward_threshold=8,        # 接近最优得分
    max_episode_steps=200,     # 超过单次最大步数限制200步未完成任务则自动终止
)

# 8. 数字猜测游戏：通过反馈猜测目标数字（适合演示Q学习）
register(
    id='GuessingGame-v0',
    entry_point='gym.envs.toy_text.guessing_game:GuessingGame',
    max_episode_steps=200,
)

# 9. 热冷游戏：根据"热/冷"反馈搜索目标位置（空间探索任务）
register(
    id='HotterColder-v0',
    entry_point='gym.envs.toy_text.hotter_colder:HotterColder',
    max_episode_steps=200,
)


### 五、MuJoCo机器人控制环境（需独立许可证）
# 基于高精度物理引擎的连续控制任务，适合复杂机器人运动研究

# 1. 2D机械臂：控制机械臂末端到达目标位置（低维控制）
register(
    id='Reacher-v1',
    entry_point='gym.envs.mujoco:ReacherEnv',
    max_episode_steps=50,     # 短步数限制
    reward_threshold=-3.75,   # 负阈值表示最小化末端与目标的距离
)

# 2. 物体推动：控制机械臂将物体推到目标位置
register(
    id='Pusher-v0',
    entry_point='gym.envs.mujoco:PusherEnv',
    max_episode_steps=100,
    reward_threshold=0.0,     # 尚未定义明确成功标准
)

# 3. 物体投掷：控制机械臂投掷物体到目标区域
register(
    id='Thrower-v0',
    entry_point='gym.envs.mujoco:ThrowerEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

# 4. 物体击打：控制机械臂击打物体到目标位置
register(
    id='Striker-v0',
    entry_point='gym.envs.mujoco:StrikerEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

# 5. 倒立摆（向上平衡）：与Classic Control中的倒立摆不同，此为向上控制
register(
    id='InvertedPendulum-v1',
    entry_point='gym.envs.mujoco:InvertedPendulumEnv',
    max_episode_steps=1000,
    reward_threshold=950.0,   # 高精度控制要求（接近1000分）
)

# 6. 二级倒立摆：控制两个连杆保持平衡（难度远高于单级）
register(
    id='InvertedDoublePendulum-v1',
    entry_point='gym.envs.mujoco:InvertedDoublePendulumEnv',
    max_episode_steps=1000,
    reward_threshold=9100.0,  # 极高的奖励阈值，要求长期稳定控制
)

# 7. 猎豹奔跑：控制四足机器人最大化奔跑速度
register(
    id='HalfCheetah-v1',
    entry_point='gym.envs.mujoco:HalfCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,  # 高速奔跑的得分要求
)

# 8. 单腿跳跃机器人：控制机器人持续跳跃（考察动态平衡）
register(
    id='Hopper-v1',
    entry_point='gym.envs.mujoco:HopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,  # 持续跳跃的得分
)

# 9. 游泳者：控制多关节机器人在水中前进
register(
    id='Swimmer-v1',
    entry_point='gym.envs.mujoco:SwimmerEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,   # 相对较低的阈值，因水中阻力较大
)

# 10. 双腿行走机器人：双足直立行走（类人运动控制）
register(
    id='Walker2d-v1',
    max_episode_steps=1000,
    entry_point='gym.envs.mujoco:Walker2dEnv',
)

# 11. 蚂蚁机器人：四足运动控制（复杂步态规划）
register(
    id='Ant-v1',
    entry_point='gym.envs.mujoco:AntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,  # 高移动效率要求
)

# 12. 类人机器人：全身关节控制（高维状态空间和动作空间）
register(
    id='Humanoid-v1',
    entry_point='gym.envs.mujoco:HumanoidEnv',
    max_episode_steps=1000,
)

# 类人机器人站立：从摔倒状态站立起来（初始状态困难）
register(
    id='HumanoidStandup-v1',
    entry_point='gym.envs.mujoco:HumanoidStandupEnv',
    max_episode_steps=1000,
)


### 六、雅达利游戏环境（Atari Environments）
# 通过Arcade Learning Environment封装的经典游戏，适合视觉强化学习研究
# 特点：状态为游戏画面像素，动作是游戏控制器输入，需处理时序依赖

# 遍历所有支持的雅达利游戏（共70款）
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
        # 生成环境名称（如将space_invaders转换为SpaceInvaders）
        name = ''.join(g.capitalize() for g in game.split('_'))
        if obs_type == 'ram':
            name = f'{name}-ram'  # RAM观测版本添加-ram后缀（使用游戏内存状态而非图像）
            
        # 处理特殊游戏的非确定性（仅ElevatorAction-ram-v0有此问题）
        nondeterministic = False
        if game == 'elevator_action' and obs_type == 'ram':
            nondeterministic = True

        # 注册基础版本（v0）：包含动作重复概率（模拟真实游戏机
