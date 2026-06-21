# BipedalWalker-v3 强化学习项目统一配置文件

# 环境配置
ENV_ID = "BipedalWalker-v3"
ENV_HARDCORE = False
RENDER_MODE = "human"

# PPO 算法超参数
LEARNING_RATE = 0.0003
GAMMA = 0.99
BATCH_SIZE = 64
UPDATE_EPOCHS = 10
TOTAL_TIMESTEPS = 1000000
N_STEPS = 2048
ENT_COEF = 0.0
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# 模型保存路径
MODEL_SAVE_PATH = "./bipedalwalker_ppo_model.zip"
LOG_DIR = "./logs/"