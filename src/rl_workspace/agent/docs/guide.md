# 使用指南

## 目录

1. [安装](#安装)
2. [快速开始](#快速开始)
3. [算法详解](#算法详解)
4. [环境配置](#环境配置)
5. [实验管理](#实验管理)
6. [可视化](#可视化)

---

## 安装

### 系统要求

- Python 3.7+
- Windows / Linux / macOS

### 依赖安装

```bash
cd agent
pip install -r requirements.txt
```

---

## 快速开始

### 1. FrozenLake 示例

```bash
python src/examples/frozen_lake_q_learning.py --epochs 5000
```

这将训练一个 Q-Learning 智能体在 FrozenLake-v1 环境中。

### 2. CartPole 示例

```bash
python src/examples/cartpole_q_learning.py --episodes 10000
```

训练一个能够平衡杆子的智能体。

### 3. PPO 训练

```bash
python src/examples/frozen_lake_ppo.py --epochs 3000
```

使用 PPO 算法训练（推荐用于复杂任务）。

### 4. 统一入口

使用统一的训练脚本：

```bash
# FrozenLake Q-Learning
python train.py --env frozen_lake --algo q_learning --epochs 5000

# CartPole Q-Learning
python train.py --env cartpole --algo q_learning --epochs 10000

# FrozenLake PPO
python train.py --env frozen_lake --algo ppo --epochs 3000

# 算法对比
python train.py --benchmark --epochs 2000
```

---

## 算法详解

### Q-Learning

经典的离策略强化学习算法。

**配置参数**:
- `alpha`: 学习率 (0.1-0.9)
- `gamma`: 折扣因子 (0.9-0.99)
- `rar`: 初始探索率 (1.0)
- `radr`: 探索率衰减 (0.995)

### SARSA

在线策略学习算法，通常比 Q-Learning 更保守但更稳定。

### DQN

深度 Q 网络，使用神经网络近似 Q 函数。

**特点**:
- 经验回放 (Experience Replay)
- 目标网络 (Target Network)
- 适合高维状态空间

### PPO

近端策略优化，目前最流行的强化学习算法之一。

**特点**:
- 限制策略更新幅度，保证训练稳定
- 适合连续动作空间
- 通常能获得更好的最终性能

---

## 环境配置

### FrozenLake

```python
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

env = gym.make(
    "FrozenLake-v1",
    desc=generate_random_map(size=8),  # 8x8 地图
    is_slippery=False  # 关闭随机性
)
```

### CartPole

```python
env = gym.make("CartPole-v1", render_mode="rgb_array")
```

### MountainCar

```python
env = gym.make("MountainCar-v0")
```

### Acrobot

```python
env = gym.make("Acrobot-v1")
```

---

## 实验管理

### 保存实验

使用 `ExperimentTracker`:

```python
from rl.experiment_manager import ExperimentTracker, ExperimentConfig

tracker = ExperimentTracker(experiment_dir="experiments")
config = ExperimentConfig(
    algorithm="q_learning",
    env_name="FrozenLake-v1",
    hyperparameters={"alpha": 0.1, "gamma": 0.99}
)

tracker.start_experiment(config)

for episode in range(epochs):
    reward = train_one_step()
    tracker.log_metric("reward", reward, step=episode)

tracker.end_experiment(success=True)
```

### 加载模型

```python
from rl.model_deployment import ModelSerializer

model, metadata = ModelSerializer.load_model("checkpoints/model.pkl")
```

---

## 可视化

### 训练曲线

自动保存到 `plots/` 目录：

- `frozen_lake_training_q_learning.png` - 训练奖励曲线
- `frozen_lake_success_rate_ppo.png` - 成功率曲线
- `algorithm_comparison.png` - 算法对比图

### 自定义可视化

```python
from rl.visualizer import TrainingVisualizer, PerformanceAnalyzer

TrainingVisualizer.plot_training_curve(
    rewards,
    title="我的实验",
    save_path="plots/my_plot.png"
)

PerformanceAnalyzer.plot_success_rate(
    rewards,
    window_size=100,
    title="成功率"
)
```

---

## 故障排除

### 导入错误

确保将 `src/` 目录添加到 Python 路径：

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
```

### 渲染问题

在无头环境（如服务器）上运行时，禁用渲染：

```python
env = gym.make("CartPole-v1", render_mode=None)
```

---

## 进阶使用

### 超参数优化

```python
from rl.experiment_manager import HyperparameterOptimizer

param_space = {
    "alpha": [0.05, 0.1, 0.2],
    "gamma": [0.9, 0.95, 0.99]
}

optimizer = HyperparameterOptimizer(param_space)
best_params = optimizer.grid_search(train_function)
```

### 分布式训练

```python
from rl.distributed_training import ParallelTrainer

trainer = ParallelTrainer(num_workers=4)
results = trainer.parallel_train(
    train_fn=train_one_worker,
    total_epochs=10000
)
```

---

## 联系方式

如有问题，请查看项目仓库或提交 Issue。
