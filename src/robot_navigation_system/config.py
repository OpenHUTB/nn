class Config:
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