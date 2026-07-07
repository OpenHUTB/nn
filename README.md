# carla_2d_deeprl

> 基于 CARLA 的 2D 深度强化学习训练框架 —— 从环境搭建到 PPO 训练，开箱即用

---

## 1. 项目概述

本项目是一个完整的 CARLA 2D 深度强化学习研究与训练框架。它在 CARLA 自动驾驶模拟器之上构建了一个俯视视角（Bird's Eye View）的强化学习环境，并提供了一套从**环境交互** → **奖励设计** → **PPO 训练** → **TensorBoard 监控** → **Episode 录制回放**的完整流水线。

### 核心特性

| 特性 | 说明 |
|------|------|
| 2D 俯视观测 | 480×480 语义分割图像，6 类标签（道路/车道线/路杆/路肩/车辆/其他） |
| 双动作空间 | 支持离散（3 动作）和连续（2 维）两种控制模式 |
| 模块化奖励 | 7 个可独立配置权重的奖励组件 |
| PPO 训练器 | 内置 CNN + Actor-Critic 网络，GAE 优势估计，完整训练循环 |
| TensorBoard | 自动记录 episode reward、各奖励分量、loss、entropy 等曲线 |
| Episode 录制 | 自动录制评估 episode，支持离线回放分析 |
| 鲁棒性设计 | 断线重连、地图加载重试、Actor 生成重试、异常清理 |

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                      train.py                           │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐    │
│  │ CNNEncoder│  │ActorCritic│  │   PPOTrainer       │    │
│  │   (4层CNN) │  │(Gaussian/ │  │ • collect_rollout  │    │
│  │           │  │ Categorical)│  │ • update (PPO)     │    │
│  └──────────┘  └──────────┘  │ • TensorBoard log   │    │
│                               │ • Episode record    │    │
│                               └──────────┬─────────┘    │
└──────────────────────────────────────────┼──────────────┘
                                           │
┌──────────────────────────────────────────┼──────────────┐
│              min_carla_env/              │              │
│  ┌────────────────┐  ┌──────────────────┐│              │
│  │  matrix_world   │  │     env.py       │              │
│  │ • 地图加载/切换  │  │ • CarlaEnv (Gym) │              │
│  │ • 车辆/传感器生成 │  │ • 观测处理       │              │
│  │ • Actor 清理    │  │ • 动作执行       │              │
│  └────────────────┘  └────────┬─────────┘              │
│                               │                         │
│  ┌────────────────┐  ┌───────┴─────────┐              │
│  │   rewards.py    │  │   recorder.py    │              │
│  │ • 7种奖励组件    │  │ • Episode 录制   │              │
│  │ • 可配置权重     │  │ • 回放与渲染     │              │
│  └────────────────┘  └─────────────────┘              │
└────────────────────────────────────────────────────────┘
                                           │
                                    ┌──────┴──────┐
                                    │   CARLA     │
                                    │  Simulator  │
                                    └─────────────┘
```

---

## 3. 环境设计

### 3.1 观测空间（Observation）

环境返回 **480×480 的语义分割图像**，经过标签映射后包含 6 个类别：

| 标签值 | 含义 | 原始 CARLA 标签 |
|--------|------|----------------|
| 0 | 无/背景 | — |
| 1 | 道路 (Road) | 7 |
| 2 | 车道线 (RoadLine) | 6 |
| 3 | 路杆 (Poles) | 5 |
| 4 | 路肩/建筑/植被等 | 1,2,3,4,8,9,11,12 |
| 5 | 车辆 (Vehicles) | 10 |

图像自动旋转使车辆始终位于画面底部中央，便于 CNN 提取空间特征。

### 3.2 动作空间（Action）

**离散模式**（`continuous=False`）：

| Action ID | 含义 | 控制量 |
|-----------|------|--------|
| 0 | 直行 (Coast) | steer=0.0, throttle=0.3 |
| 1 | 左转 (Turn Left) | steer=-0.5, throttle=0.3 |
| 2 | 右转 (Turn Right) | steer=0.5, throttle=0.3 |

**连续模式**（`continuous=True`，默认）：

| 维度 | 范围 | 含义 |
|------|------|------|
| `action[0]` | [-1, 1] | 转向角（负=左，正=右） |
| `action[1]` | [-1, 1] | 油门/刹车（正=油门，负=刹车） |

### 3.3 终止条件

Episode 在以下任一条件触发时终止：

- 达到最大步数（`max_step`，默认 90000）
- 发生碰撞
- 压实线（Solid）或无标线（NONE）车道线
- 驶入人行道（Sidewalk）
- 连续卡死超过 20 步
- 累计奖励低于 -1000

---

## 4. 奖励函数设计

奖励系统由 [rewards.py](min_carla_env/rewards.py) 中的 `RewardCalculator` 类实现，包含 **7 个可独立配置权重的奖励组件**：

| 组件 | 默认权重 | 计算逻辑 |
|------|---------|---------|
| `lane_center` | **1.0** | 距离车道中心 < 0.5m 奖励 +0.5，否则按 $-\exp(d)$ 惩罚 |
| `speed` | **0.5** | 接近目标速度 30km/h（±5km/h）奖励 +0.3，偏差越大惩罚越重 |
| `collision` | **-50.0** | 发生碰撞时一次性惩罚 |
| `lane_invasion` | **-20.0** | 压实线或无标线路段时惩罚 |
| `stuck` | **-10.0** | 卡死时惩罚 |
| `progress` | **0.1** | 按每步行驶距离给予正向奖励，鼓励持续前进 |
| `steering_smoothness` | **-0.05** | 惩罚相邻两步转向角的大幅变化，鼓励平滑操控 |

### 自定义权重

```python
from min_carla_env.rewards import RewardCalculator

custom_weights = {
    "lane_center": 2.0,     # 更重视车道保持
    "speed": 0.3,
    "collision": -100.0,    # 更严厉的碰撞惩罚
    "lane_invasion": -30.0,
    "stuck": -10.0,
    "progress": 0.2,
    "steering_smoothness": -0.1,
}

env = CarlaEnv(client, config, reward_weights=custom_weights)
```

---

## 5. 核心模块详解

### 5.1 matrix_world.py — 世界管理

`MatrixWorld` 类负责与 CARLA 模拟器的底层交互：

- **地图管理**：加载/切换 Town01~Town07 等城市场景，支持重试机制
- **车辆生成**：在随机出生点或路口附近生成 Tesla Model 3
- **传感器挂载**：RGB 相机、语义分割相机（俯视 2D 视角）、碰撞传感器、车道入侵传感器
- **资源清理**：`clean_world()` 彻底销毁所有 Actor 并停止传感器监听

### 5.2 env.py — Gym 环境封装

`CarlaEnv` 类遵循 OpenAI Gym 接口规范：

- `reset()` → 返回初始观测
- `step(action)` → 返回 `(obs, reward, done, info)`
- `close()` → 释放所有资源

**鲁棒性设计**：
- 初始化失败自动重连 CARLA 客户端
- Actor 生成失败最多重试 5 次
- 每次 `reset()` 彻底清理并重建所有 Actor

### 5.3 recorder.py — Episode 录制

`EpisodeRecorder` 和 `EpisodeReplayer` 提供完整的录制回放能力：

```python
from min_carla_env.recorder import EpisodeRecorder, EpisodeReplayer

# 录制
recorder = EpisodeRecorder(save_dir="my_recordings")
recorder.record_step(obs, action, reward, info)
recorder.save(episode_id=1)

# 回放
replayer = EpisodeReplayer("my_recordings/episode_1")
obs, action, reward, done = replayer.step()
frame = replayer.render_frame(idx=0)  # 渲染指定帧
```

### 5.4 train.py — PPO 训练器

完整的 PPO（Proximal Policy Optimization）实现：

**网络结构**：
```
输入: 480×480×1 语义分割图
  ↓
Conv2d(1→32, kernel=8, stride=4) + ReLU
  ↓
Conv2d(32→64, kernel=4, stride=2) + ReLU
  ↓
Conv2d(64→64, kernel=3, stride=1) + ReLU
  ↓
Flatten → Linear(→256) + ReLU
  ↓
┌──────────────┐  ┌──────────────┐
│  Actor Head  │  │  Critic Head │
│ Linear(256→2)│  │ Linear(256→1)│
│ + log_std    │  │  (Value)     │
└──────────────┘  └──────────────┘
```

**PPO 核心参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lr` | 3e-4 | Adam 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE 平滑参数 |
| `clip_ratio` | 0.2 | PPO 裁剪范围 |
| `ent_coef` | 0.01 | 熵正则化系数 |
| `update_epochs` | 10 | 每轮数据更新次数 |
| `batch_size` | 64 | 小批量大小 |
| `buffer_capacity` | 2048 | 经验池容量 |

---

## 6. 快速开始

### 6.1 环境要求

- Python 3.7+
- CARLA 0.9.x 服务端运行中
- CUDA（可选，用于 GPU 训练加速）

### 6.2 安装依赖

```bash
cd carla_2d_deeprl
pip install -r requirements.txt
```

### 6.3 启动 CARLA 服务端

```bash
# Linux
./CarlaUE4.sh -quality-level=Low -RenderOffScreen

# Windows
CarlaUE4.exe -quality-level=Low -RenderOffScreen
```

### 6.4 测试环境

```bash
python test_env.py
```

### 6.5 开始训练

```bash
# 连续动作空间训练（默认）
python train.py --town Town02 --total-steps 500000

# 离散动作空间训练
python train.py --discrete --total-steps 500000

# 无渲染模式（更快）
python train.py --town Town02 --total-steps 500000
# 在 world_config 中设置 render=False
```

### 6.6 评估模型

```bash
python train.py --evaluate --load checkpoints/model_final.pt --eval-episodes 10
```

### 6.7 断点续训

```bash
python train.py --load checkpoints/model_100000.pt --total-steps 500000
```

### 6.8 查看训练曲线

```bash
tensorboard --logdir runs
```

浏览器打开 `http://localhost:6006` 即可查看：
- `episode/reward` — Episode 总奖励
- `episode/length` — Episode 步数
- `reward/r_lane_center` — 车道中心奖励分量
- `reward/r_speed` — 速度奖励分量
- `train/policy_loss` — 策略损失
- `train/value_loss` — 价值损失
- `train/entropy` — 策略熵

---

## 7. 配置说明

### 环境配置（CONFIG）

```python
CONFIG = {
    "width": 480,           # 观测图像宽度
    "height": 480,          # 观测图像高度
    "max_step": 90000,      # 单 episode 最大步数
    "render": True,         # 是否渲染
    "continuous": True,     # 是否使用连续动作空间
    "target_speed": 30.0,   # 目标速度 (km/h)
}
```

### 世界配置（world_config）

```python
world_config = {
    "im_width": 480.0,      # 传感器图像宽度
    "im_height": 480.0,     # 传感器图像高度
    "render": True,         # 是否渲染
    "weather": None,        # 天气参数（carla.WeatherParameters）
    "fast": True,           # 快速仿真模式（fixed_delta_seconds=0.05）
    "town": "Town02",       # 地图名称
}
```

---

## 8. 文件结构

```
carla_2d_deeprl/
├── README.md                          # 项目文档（本文件）
├── requirements.txt                   # Python 依赖
├── train.py                           # PPO 训练入口（含 CNN 网络、训练循环、TensorBoard）
├── test_env.py                        # 环境功能测试脚本
│
├── checkpoints/                       # 模型保存目录（训练自动创建）
│   ├── model_50000.pt
│   └── model_final.pt
│
├── runs/                              # TensorBoard 日志目录（训练自动创建）
│   └── recordings/                    # Episode 录制文件
│       └── episode_*/
│           ├── data.npz               # (obs, action, reward) 数组
│           └── summary.json           # episode 摘要
│
└── min_carla_env/                     # 核心环境包
    ├── __init__.py
    ├── matrix_world.py                # CARLA 世界管理（地图/车辆/传感器/Actor 清理）
    ├── env.py                         # Gym 环境封装（CarlaEnv: reset/step/close）
    ├── rewards.py                     # 模块化奖励函数（7 组件，可配置权重）
    └── recorder.py                    # Episode 录制与回放工具
```

---

## 9. 设计亮点

### 9.1 鲁棒性

- **断线重连**：CARLA 客户端断开时自动重连（最多 3 次）
- **地图加载重试**：地图加载失败自动重试（最多 2 次）
- **Actor 生成重试**：车辆/传感器生成失败自动重试（最多 5 次）
- **异常清理**：任何初始化失败都会触发 `clean_world()` 清理残留 Actor

### 9.2 模块化

- 奖励函数与环境解耦，`RewardCalculator` 可独立测试和替换
- 录制回放与训练逻辑分离，`EpisodeRecorder` 可单独使用
- PPO 训练器与具体环境解耦，只需满足 Gym 接口即可替换

### 9.3 可观测性

- TensorBoard 实时监控训练曲线
- `info` 字典返回奖励分量分解，便于分析各组件贡献
- Episode 录制支持离线回放，直观评估策略表现

---

## 10. 参考

- 原始参考项目：[mcemilg/min-carla-env](https://github.com/mcemilg/min-carla-env)
- CARLA 官方文档：[carla.readthedocs.io](https://carla.readthedocs.io/)
- PPO 论文：[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- GAE 论文：[High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)