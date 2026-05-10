import carla
from min_carla_env.env import CarlaEnv

# 1. 初始化CARLA客户端
client = carla.Client('localhost', 2000)  # 默认端口2000，根据你的CARLA服务调整
client.set_timeout(10.0)

# 2. 配置参数（对应CONFIG字典）
config = {
    "width": 480,
    "height": 480,
    "max_step": 90000,
    "render": False  # 不渲染（对应你想要的show=False）
}

# 3. world配置（对应MatrixWorld的参数）
world_config = {
    "fast": True,     # 快速模式
    "town": "Town02"  # 可选，指定地图
}

# 4. 初始化CarlaEnv
env = CarlaEnv(
    client=client,    # 必须传client
    config=config,    # 渲染/分辨率等配置
    world_config=world_config,  # 快速模式/地图等配置
    debug=False,      # 可选：是否输出调试图片
    demo=False        # 可选：是否演示模式
)

# 测试环境重置和步进
obs = env.reset()
print(f"重置后观测形状: {obs.shape}")

# 测试执行动作（0=直行，1=左转，2=右转）
obs, reward, done, info = env.step(0)
print(f"动作执行后 - 奖励: {reward}, 是否结束: {done}")

# 清理环境
env.mw.clean_world()