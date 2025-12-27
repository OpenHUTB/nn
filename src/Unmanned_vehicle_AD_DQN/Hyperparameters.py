# Hyperparameters.py
# 深度强化学习超参数配置 - 优化避障性能

DISCOUNT = 0.98
# 提高未来奖励的重要性，让智能体更关注长期安全

FPS = 60

MEMORY_FRACTION = 0.6
# 增加GPU内存分配

REWARD_OFFSET = -100

MIN_REPLAY_MEMORY_SIZE = 4_000
# 增加最小回放缓冲区大小

REPLAY_MEMORY_SIZE = 15_000
# 增加回放缓冲区容量

MINIBATCH_SIZE = 32
# 减少批次大小以避免内存不足

PREDICTION_BATCH_SIZE = 1

TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4

EPISODES = 100  # 减少训练轮次以加快训练速度
# 增加训练轮次以获得更好的避障性能

SECONDS_PER_EPISODE = 70
# 稍微增加每轮时间

MIN_EPSILON = 0.02
# 稍微提高最小探索率

EPSILON = 1.0

EPSILON_DECAY = 0.995
# 减缓探索率衰减

MODEL_NAME = "YY_Enhanced_ObstacleAvoidance"
# 更新模型名称

MIN_REWARD = 8
# 提高良好经验的最低奖励阈值

UPDATE_TARGET_EVERY = 25
# 减缓目标网络更新频率

AGGREGATE_STATS_EVERY = 10

SHOW_PREVIEW = False

IM_WIDTH = 320

IM_HEIGHT = 240

SLOW_COUNTER = 330

LOW_REWARD_THRESHOLD = -2

SUCCESSFUL_THRESHOLD = 5
# 提高成功阈值

LEARNING_RATE = 0.00003
# 降低学习率以获得更稳定训练

# PER (优先经验回放) 参数
PER_ALPHA = 0.7
# 提高优先级程度

PER_BETA_START = 0.5
# 提高重要性采样起始值

PER_BETA_FRAMES = 120000
# 增加beta线性增长的帧数

# 训练策略参数
USE_CURRICULUM_LEARNING = True

USE_MULTI_OBJECTIVE = True

USE_IMITATION_LEARNING = True
# 启用模仿学习

USE_ATTENTION_MECHANISM = True
# 启用注意力机制

USE_ENHANCED_MODEL = True
# 启用增强版模型

# 避障优化参数
OBSTACLE_AVOIDANCE_WEIGHT = 0.15
NEAR_MISS_REWARD = 2.0
SAFE_DISTANCE_REWARD = 1.0
COLLISION_PENALTY = -15.0
DANGER_ZONE_PENALTY = -5.0

# 课程学习参数
CURRICULUM_STAGES = 6
# 增加课程学习阶段

CURRICULUM_SUCCESS_THRESHOLDS = [0.4, 0.5, 0.6, 0.7, 0.75, 0.8]
# 调整成功率阈值

# 新增：障碍物检测参数
OBSTACLE_DETECTION_MODE = 'advanced'  # 'basic', 'advanced', 'none'
OBSTACLE_WARNING_THRESHOLDS = [20.0, 12.0, 8.0, 5.0]  # 警告级别距离阈值
BUILDING_PROXIMITY_THRESHOLD = 15.0  # 建筑物接近阈值（米）

# 新增：奖励函数参数
REWARD_CONFIG = {
    'base_survival': 0.1,
    'perfect_lane_keeping': 0.8,
    'good_lane_keeping': 0.4,
    'acceptable_lane_keeping': 0.1,
    'lane_deviation_penalty': 0.3,
    'ideal_speed_range': (18, 38),
    'ideal_speed_reward': 0.4,
    'low_speed_reward': 0.2,
    'high_speed_penalty': 0.3,
    'steer_smoothness_factor': 0.2,
    'progress_weight': 0.3,
    'destination_bonus': 25,
    'off_road_penalty': 5,
    'time_penalty_threshold': 60  # 秒
}

# 新增：动作调整参数
ACTION_ADJUSTMENT = {
    'speed_factor_min': 0.4,
    'speed_factor_max': 1.0,
    'emergency_brake_distance': 6.0,
    'max_consecutive_steer': 3,
    'high_speed_steer_threshold': 30
}