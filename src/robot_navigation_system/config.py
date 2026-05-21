class Config:
    """机器人导航系统的配置类。

    Attributes:
        ROBOT_RADIUS: 机器人半径（米）
        MAX_SPEED: 机器人最大线速度（米/秒）
        MAX_ANGULAR_SPEED: 机器人最大角速度（弧度/秒）
        MAP_WIDTH: 地图宽度（米）
        MAP_HEIGHT: 地图高度（米）
        GRID_SIZE: 网格大小（米），用于障碍物膨胀
        LIDAR_ANGLES: 激光雷达角度分辨率（度）
        LIDAR_RANGE: 激光雷达最大探测距离（米）
        LIDAR_NOISE: 激光雷达噪声标准差
        STATE_SIZE: DQN 状态空间维度（激光雷达360度 + 4维目标信息）
        ACTION_SIZE: DQN 动作空间维度
        LEARNING_RATE: DQN 学习率
        GAMMA: 折扣因子
        EPSILON_START: 初始探索率
        EPSILON_END: 最终探索率
        EPSILON_DECAY: 探索率衰减系数
        BATCH_SIZE: 经验回放批次大小
        MEMORY_SIZE: 经验回放缓冲区容量
        TARGET_UPDATE: 目标网络更新频率（步数）
        EPISODES: 训练回合数
        MAX_STEPS: 每回合最大步数
        REWARD_GOAL: 到达目标奖励
        REWARD_COLLISION: 碰撞惩罚
        REWARD_STEP: 每步基础惩罚
        VISUALIZE: 是否启用可视化
        PLOT_INTERVAL: 可视化更新间隔（回合数）
        SAVE_RESULTS: 是否保存训练结果
        RESULT_DIR: 结果保存目录
    """
    # 机器人参数
    ROBOT_RADIUS = 0.3
    MAX_SPEED = 1.0
    MAX_ANGULAR_SPEED = 1.57
    
    # 环境参数
    MAP_WIDTH = 20
    MAP_HEIGHT = 20
    GRID_SIZE = 0.5
    
    # 激光雷达参数
    LIDAR_ANGLES = 360
    LIDAR_RANGE = 10.0
    LIDAR_NOISE = 0.05
    
    # DQN参数
    STATE_SIZE = 360 + 4  
    ACTION_SIZE = 5       
    LEARNING_RATE = 0.001
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    BATCH_SIZE = 64
    MEMORY_SIZE = 100000
    TARGET_UPDATE = 10
    
    # 训练参数
    EPISODES = 500
    MAX_STEPS = 500
    REWARD_GOAL = 100.0
    REWARD_COLLISION = -50.0
    REWARD_STEP = -0.1
    
    # 可视化参数
    VISUALIZE = True
    PLOT_INTERVAL = 10
    SAVE_RESULTS = True
    RESULT_DIR = './results'
    
    # 目标点
    TARGET_POSITION = (15.0, 15.0)
    START_POSITION = (2.0, 2.0)
    
    # 障碍物参数
    OBSTACLE_COUNT = 15
    OBSTACLE_MIN_RADIUS = 0.5
    OBSTACLE_MAX_RADIUS = 1.5