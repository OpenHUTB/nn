# 机器人导航系统模块

基于深度强化学习（DQN）的机器人自主导航系统，支持激光雷达环境感知、实时避障和路径规划。

## 功能特性

- ✅ 激光雷达环境感知（360度扫描）
- ✅ DQN深度强化学习路径规划
- ✅ 实时避障系统
- ✅ 丰富的可视化结果
- ✅ 训练历史记录

## 项目结构

```text
robot_navigation_system/
├── config.py              # 配置文件
├── environment.py         # 导航环境模拟
├── dqn_agent.py           # DQN强化学习代理
├── visualization.py       # 可视化模块
├── train.py               # 训练脚本
├── test.py                # 测试脚本
├── main.py                # 主入口
├── requirements.txt       # 依赖列表
├── results/               # 可视化结果目录
│   ├── navigation_episode_*.png  # 导航截图
│   ├── training_history.png      # 训练历史图
│   └── lidar_heatmap.png         # 激光雷达热力图
└── README.md              # 项目说明
```

## 快速开始

```bash
# 进入项目目录
cd src/robot_navigation_system

# 安装依赖
pip install -r requirements.txt

# 训练模型
python train.py

# 测试模型
python test.py
```

## 配置参数

在`config.py`中可以调整以下参数：

| 参数类别 | 说明 |
|---------|------|
| 机器人参数 | 半径、最大速度、角速度 |
| 环境参数 | 地图尺寸、网格大小 |
| 激光雷达 | 角度数、探测范围、噪声 |
| DQN参数 | 学习率、折扣因子、ε衰减 |
| 训练参数 | 轮数、最大步数、奖励设置 |

## 可视化结果

项目运行后会在`results/`目录生成：

1. **navigation_episode_X.png** - 每10轮训练的导航截图
2. **training_history.png** - 训练奖励、距离、步数历史图表
3. **lidar_heatmap.png** - 激光雷达数据极坐标热力图
4. **test_navigation_result.png** - 测试导航结果

## 控制动作

| 动作 | 说明 |
|------|------|
| 0 | 前进 |
| 1 | 左转前进 |
| 2 | 右转前进 |
| 3 | 原地左转 |
| 4 | 原地右转 |

## 奖励机制

- ✅ 到达目标: +100
- ❌ 碰撞障碍物: -50
- ⏭️ 每步惩罚: -0.1

## 技术栈

- Python 3.8+
- PyTorch 2.2+
- NumPy
- Matplotlib
- OpenCV

## 参考文献

- Mnih, V., et al. "Human-level control through deep reinforcement learning." Nature, 2015.
- Sutton, R. S., & Barto, A. G. "Reinforcement learning: An introduction." MIT press, 2018.