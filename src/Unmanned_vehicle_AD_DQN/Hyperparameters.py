# Hyperparameters.py
# 深度强化学习超参数配置

DISCOUNT = 0.97
# 未来奖励的折扣因子 - 提高未来奖励的重要性

FPS = 60
# 模拟环境的帧率

MEMORY_FRACTION = 0.35
# GPU内存分配比例

REWARD_OFFSET = -100
# 停止模拟的奖励阈值

MIN_REPLAY_MEMORY_SIZE = 3_000
# 开始训练前经验回放缓冲区的最小大小 - 增加以获得更稳定训练

REPLAY_MEMORY_SIZE = 10_000
# 经验回放缓冲区的最大容量

MINIBATCH_SIZE = 64
# 每次训练从经验回放中采样的经验数量 - 增加批次大小

PREDICTION_BATCH_SIZE = 1
# 预测阶段使用的批次大小

TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
# 训练阶段使用的批次大小

EPISODES = 800  # 从1000减少到800，因为减少了无行人阶段
# 智能体训练的总轮次数

SECONDS_PER_EPISODE = 60
# 每轮训练的秒数

MIN_EPSILON = 0.01
# 最小探索率

EPSILON = 1.0
# 初始探索率

EPSILON_DECAY = 0.998
# 探索率的衰减率 - 减缓衰减速度

MODEL_NAME = "YY_Optimized"
# 训练模型的名称标识

MIN_REWARD = 5
# 被认为是"良好"或"积极"经验的最小奖励值

UPDATE_TARGET_EVERY = 20
# 目标网络更新的频率 - 增加以获得更稳定训练

AGGREGATE_STATS_EVERY = 10
# 计算和聚合统计信息（如平均得分、奖励）的频率

SHOW_PREVIEW = False
# 是否显示预览窗口 - 测试时设为False以显示CARLA主窗口

IM_WIDTH = 640
# 预览或模拟中捕获图像的宽度

IM_HEIGHT = 480
# 预览或模拟中捕获图像的高度

SLOW_COUNTER = 330
# 慢速计数器阈值

LOW_REWARD_THRESHOLD = -2
# 低奖励阈值

SUCCESSFUL_THRESHOLD = 3
# 成功阈值

LEARNING_RATE = 0.00005
# 优化器的学习率 - 降低以获得更稳定训练

# PER (优先经验回放) 参数
PER_ALPHA = 0.6
# 优先级程度 (0 = 均匀采样, 1 = 完全优先级)

PER_BETA_START = 0.4
# 重要性采样权重起始值

PER_BETA_FRAMES = 100000
# beta线性增长的帧数

# Dueling DQN 参数
USE_DUELING = True
# 是否使用Dueling DQN架构

USE_PER = True
# 是否使用优先经验回放