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

MIN_REPLAY_MEMORY_SIZE = 1_000  # 减少预热样本数，让训练更快开始
# 开始训练前经验回放缓冲区的最小大小

REPLAY_MEMORY_SIZE = 15_000  # 增加经验回放容量
# 经验回放缓冲区的最大容量

MINIBATCH_SIZE = 128  # 增加批次大小
# 每次训练从经验回放中采样的经验数量

PREDICTION_BATCH_SIZE = 1
# 预测阶段使用的批次大小

TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
# 训练阶段使用的批次大小

EPISODES = 500  # 增加训练轮次
# 智能体训练的总轮次数

SECONDS_PER_EPISODE = 60
# 每轮训练的秒数

MIN_EPSILON = 0.05  # 增加最小探索率，保持一定的探索
# 最小探索率

EPSILON = 1.0
# 初始探索率

EPSILON_DECAY = 0.995  # 减缓衰减速度
# 探索率的衰减率

MODEL_NAME = "YY_Optimized_v2"
# 训练模型的名称标识

MIN_REWARD = -100  # 降低阈值，更容易保存模型
# 被认为是"良好"或"积极"经验的最小奖励值

UPDATE_TARGET_EVERY = 15  # 更频繁地更新目标网络
# 目标网络更新的频率

AGGREGATE_STATS_EVERY = 5  # 更频繁地记录统计信息
# 计算和聚合统计信息（如平均得分、奖励）的频率

SHOW_PREVIEW = False
# 是否显示预览窗口 - 测试时设为False以显示CARLA主窗口

IM_WIDTH = 160  # 减少图像尺寸，加快处理速度
# 预览或模拟中捕获图像的宽度

IM_HEIGHT = 120  # 减少图像尺寸，加快处理速度
# 预览或模拟中捕获图像的高度

SLOW_COUNTER = 330
# 慢速计数器阈值

LOW_REWARD_THRESHOLD = -2
# 低奖励阈值

SUCCESSFUL_THRESHOLD = 3
# 成功阈值

LEARNING_RATE = 0.0001  # 增加学习率，加速收敛
# 优化器的学习率

# PER (优先经验回放) 参数
PER_ALPHA = 0.7  # 增加优先级影响
# 优先级程度 (0 = 均匀采样, 1 = 完全优先级)

PER_BETA_START = 0.5
# 重要性采样权重起始值

PER_BETA_FRAMES = 50000  # 减少beta增长时间
# beta线性增长的帧数

# 训练策略参数
USE_CURRICULUM_LEARNING = True
# 是否使用课程学习

USE_MULTI_OBJECTIVE = True
# 是否使用多目标优化

USE_IMITATION_LEARNING = False
# 是否使用模仿学习（预训练）

# 多目标优化权重（这些权重会自动调整）
SAFETY_WEIGHT = 0.50  # 提高安全权重
EFFICIENCY_WEIGHT = 0.20
COMFORT_WEIGHT = 0.15
RULE_FOLLOWING_WEIGHT = 0.15

# 课程学习参数
CURRICULUM_STAGES = 5
# 课程学习阶段数量

CURRICULUM_SUCCESS_THRESHOLDS = [0.3, 0.5, 0.7, 0.85, 0.9]
# 每个阶段进入下一阶段所需的成功率阈值

# 新增：反应时间参数
REACTION_TIME_PENALTY = 0.1  # 对反应迟钝的惩罚系数
PROACTIVE_REWARD = 0.5  # 对提前行动的奖励