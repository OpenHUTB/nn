from stable_baselines3 import SAC
from env_mujoco import make_env
import os

# 1. 初始化环境
env = make_env()

# 2. 模型路径（把你的模型文件改名为 my_humanoid_model.zip）
model_path = "my_humanoid_model.zip"

if os.path.exists(model_path):
    # 加载已有的模型
    model = SAC.load(model_path, env=env)
    print("成功加载现有模型！")
else:
    # 如果没找到，则新建（不推荐，因为重头练很慢）
    model = SAC("MlpPolicy", env, verbose=1)

# 3. 稍微微调一下（比如练 5000 步，确保在你的电脑上能跑通）
model.learn(total_timesteps=5000)

# 4. 保存
model.save("humanoid_final_walking")  
