import torch


class DQNConfig:
    """DQN 超参数配置"""

    # 动作空间: 前后左右上下悬停
    ACTION_DIM = 7

    # 状态空间: 位置(3) + 速度(3) + 碰撞标志(1)
    STATE_DIM = 7

    # 网络参数
    HIDDEN_DIM = 128
    LEARNING_RATE = 1e-3

    # 训练参数
    GAMMA = 0.99              # 折扣因子
    EPSILON_START = 1.0       # 初始探索率
    EPSILON_END = 0.01        # 最终探索率
    EPSILON_DECAY = 0.995     # 探索率衰减
    BATCH_SIZE = 64
    BUFFER_SIZE = 10000
    TARGET_UPDATE = 10        # 目标网络更新频率

    # 训练设置
    MAX_EPISODES = 200
    MAX_STEPS = 300

    # 奖励设置
    REWARD_ALIVE = 1.0        # 每步存活奖励
    REWARD_COLLISION = -100   # 碰撞惩罚
    REWARD_FORWARD = 0.1      # 向前移动奖励
    REWARD_STEP = -0.01       # 时间成本

    # 设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 动作映射: 动作ID -> (vx, vy, vz)
    ACTION_VELOCITIES = {
        0: (2.0, 0.0, 0.0),    # 前进
        1: (-2.0, 0.0, 0.0),   # 后退
        2: (0.0, -2.0, 0.0),   # 左移
        3: (0.0, 2.0, 0.0),    # 右移
        4: (0.0, 0.0, -2.0),   # 上升 (Z轴向下为正)
        5: (0.0, 0.0, 2.0),    # 下降
        6: (0.0, 0.0, 0.0),    # 悬停
    }
