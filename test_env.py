# 单独测试BipedalWalker-v3环境是否正常可用
import gymnasium as gym

# 创建平地双足行走环境
env = gym.make("BipedalWalker-v3", render_mode="human")
# 重置环境
observation, info = env.reset()

# 随机动作运行500步，测试环境是否正常
for step in range(500):
    random_action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(random_action)
    if done or truncated:
        env.reset()

# 关闭环境
env.close()
print("✅ 环境测试运行完毕，无报错")