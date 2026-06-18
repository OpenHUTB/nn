# AirSim Drone Controller

基于 Microsoft AirSim 的无人机控制器项目，支持多种控制方式和强化学习训练。

## 项目概述

本项目提供了一个灵活的无人机控制框架，可用于 AirSim 仿真环境中的无人机操控和强化学习研究。项目支持键盘手动控制、随机游走、视觉飞行以及基于 DQN 的强化学习智能体。

## 功能特性

- **多种控制方式**：键盘控制、随机游走、视觉飞行、DQN强化学习
- **Gym 风格环境**：提供标准的强化学习环境接口
- **完整的 DQN 实现**：包含经验回放、目标网络、Double DQN 等特性
- **模块化设计**：易于扩展和定制
- **模拟模式支持**：无需 AirSim 也可运行测试

## 项目结构

```
airsim_controller/
├── agents/                  # 智能体实现
│   ├── agent.py            # 智能体基类
│   ├── keyboard_controller.py  # 键盘控制器
│   ├── random_walker.py    # 随机游走智能体
│   ├── vision_flyer.py     # 视觉飞行智能体
│   └── dqn_agent.py        # DQN 强化学习智能体
├── agents/dqn/             # DQN 组件
│   ├── config.py           # DQN 配置
│   ├── network.py          # Q 网络定义
│   └── replay_buffer.py    # 经验回放缓冲区
├── environments/           # 环境封装
│   └── drone_env.py        # 无人机 Gym 风格环境
├── client/                 # 客户端实现
│   ├── airsim_client.py    # AirSim 客户端基类
│   └── drone_client.py     # 无人机客户端
├── airsim_utils/           # 工具函数
│   └── __init__.py         # 状态解析、碰撞检测等
├── models/                 # 模型保存目录
│   └── dqn_model.pth       # 训练后的 DQN 模型
└── requirements.txt        # 依赖包列表
```

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- `airsim` - AirSim 客户端库
- `numpy` - 数值计算
- `torch` - 深度学习框架
- `keyboard` - 键盘输入
- `Pillow` - 图像处理

## 快速开始

### 键盘控制

```python
from client.drone_client import DroneClient
from agents.keyboard_controller import KeyboardController

# 创建客户端和控制器
client = DroneClient(interval=5, root_path='./')
controller = KeyboardController(client, move_type='velocity')

# 开始控制 (按 ESC 退出)
controller.run()
```

**键盘控制说明：**
- `W` - 向前移动
- `S` - 向后移动
- `A` - 向左转向
- `D` - 向右转向
- `Q` - 向上移动
- `E` - 向下移动
- `空格` - 停止移动
- `ESC` - 退出控制

### 随机游走

```python
from client.drone_client import DroneClient
from agents.random_walker import RandomWalker

client = DroneClient(interval=5, root_path='./')
agent = RandomWalker(client, move_type='velocity', random_range=(-2, 2))
agent.run(loop_cnt=100)
```

### DQN 训练

```python
from client.drone_client import DroneClient
from agents.dqn_agent import DQNAgent
from agents.dqn.config import DQNConfig

# 创建配置
config = DQNConfig()

# 创建智能体
client = DroneClient(interval=5, root_path='./')
agent = DQNAgent(client, move_type='velocity', config=config)

# 训练
agent.train(episodes=200, save_path='./models')
```

### 加载模型并测试

```python
from client.drone_client import DroneClient
from agents.dqn_agent import DQNAgent
from agents.dqn.config import DQNConfig

client = DroneClient(interval=5, root_path='./')
agent = DQNAgent(client, move_type='velocity')

# 加载训练好的模型
agent.load('./models/dqn_model.pth')

# 测试
agent.run(episodes=10)
```

## DQN 配置说明

在 `agents/dqn/config.py` 中可以调整以下超参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ACTION_DIM` | 7 | 动作空间维度（前后左右上下悬停） |
| `STATE_DIM` | 7 | 状态空间维度（位置3+速度3+碰撞1） |
| `HIDDEN_DIM` | 128 | 隐藏层维度 |
| `LEARNING_RATE` | 1e-3 | 学习率 |
| `GAMMA` | 0.99 | 折扣因子 |
| `EPSILON_START` | 1.0 | 初始探索率 |
| `EPSILON_END` | 0.01 | 最终探索率 |
| `BATCH_SIZE` | 64 | 批次大小 |
| `BUFFER_SIZE` | 10000 | 经验回放缓冲区大小 |
| `MAX_EPISODES` | 200 | 最大训练回合数 |
| `MAX_STEPS` | 300 | 每回合最大步数 |

## 动作空间

DQN 智能体支持 7 个离散动作：

| 动作 ID | 动作 | 速度 (m/s) |
|---------|------|------------|
| 0 | 前进 | (2.0, 0.0, 0.0) |
| 1 | 后退 | (-2.0, 0.0, 0.0) |
| 2 | 左移 | (0.0, -2.0, 0.0) |
| 3 | 右移 | (0.0, 2.0, 0.0) |
| 4 | 上升 | (0.0, 0.0, -2.0) |
| 5 | 下降 | (0.0, 0.0, 2.0) |
| 6 | 悬停 | (0.0, 0.0, 0.0) |

## 奖励设计

当前奖励函数包含以下组成部分：
- **存活奖励**：每步 +1.0
- **碰撞惩罚**：-100
- **前进奖励**：动作为前进时 +0.1
- **时间成本**：每步 -0.01

## 环境要求

- Python 3.7+
- Microsoft AirSim (用于实际仿真)
- CUDA (可选，用于 GPU 加速)

## 注意事项

1. **坐标系统**：项目使用 NED (North-East-Down) 坐标系，Z 轴向下为正
2. **碰撞检测**：所有智能体都内置了碰撞检测功能
3. **模拟模式**：未安装 AirSim 时，项目会自动使用模拟模式进行测试

## 开发路线

- [ ] 支持 Dueling DQN 网络
- [ ] 添加更多奖励函数选项
- [ ] 支持多智能体训练
- [ ] 添加可视化工具
- [ ] 支持 PPO/SAC 等其他强化学习算法

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
