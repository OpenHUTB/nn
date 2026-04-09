# 无人机 Reinforcement Learning（强化学习）路径规划

使用 Reinforcement Learning（强化学习）与 AirSim 模拟器实现无人机自主路径规划。

## 概述

本项目实现了一个端到端的无人机路径规划 Reinforcement Learning（强化学习）流程：
- **环境**：为 AirSim 多旋翼无人机定制的 OpenAI Gym 封装
- **算法**：Stable Baselines3（PPO/DQN）
- **观测**：深度相机图像（84x84 灰度）
- **动作**：7 个离散动作（±X/Y/Z + 悬停）
- **奖励**：多任务奖励（距离惩罚 + 碰撞惩罚 + 成功奖励）

## 项目结构

说明：项目名 `drone_rl_planner` 中的 `rl` 是 `Reinforcement Learning` 的缩写。

```
hh/
├── src/
│   ├── envs/                    # 环境封装
│   │   ├── base_drone_env.py   # AirSim Gym 环境
│   │   └── wrappers.py          # 帧堆叠、归一化
│   ├── agents/                  # 模型实现
│   ├── training/                # 训练流程
│   │   ├── train.py             # 主训练脚本
│   │   ├── callbacks.py         # 自定义回调
│   │   └── config.yaml          # 超参数
│   ├── evaluation/              # 评估与可视化
│   │   ├── evaluate.py          # 模型评估
│   │   ├── trajectory_vis.py    # 3D 轨迹可视化
│   │   └── metrics.py           # 性能指标
│   └── utils/                   # 工具函数
│       ├── airsim_utils.py      # AirSim 接口
│       └── reward_shaper.py     # 奖励函数
├── data/
│   ├── logs/                    # TensorBoard 日志
│   ├── checkpoints/             # 模型检查点
│   ├── best_model/              # 最优模型存储
│   └── results/                 # 评估结果
├── tests/                       # 单元测试
├── config.yaml                  # 训练配置
└── train.py                     # 快速启动脚本
```

## 安装

### 前置条件
- Python >= 3.10
- 本地运行的 AirSim 模拟器
- 支持 CUDA 的 GPU（可选但推荐）

### 安装步骤

1. 克隆项目并安装依赖：
```bash
cd drone_rl_planner
pip install -r requirements.txt
```

或者使用 pip 可编辑安装：
```bash
pip install -e ".[dev]"
```

2. 验证安装：
```bash
python -c "import gymnasium; import stable_baselines3; print('✓ Ready')"
```

## 快速开始

### 1. 验证环境

```bash
python -m pytest tests/test_env.py -v
```

### 2. 训练模型

```bash
# 默认训练（读取 config.yaml）
python -m src.training.train

# 自定义训练步数
python -m src.training.train --timesteps 500000

# 从检查点继续训练
python -m src.training.train --load ./data/checkpoints/drone_model_100000_steps.zip
```

### 3. 评估模型

```bash
# 评估已训练模型
python -m src.evaluation.evaluate ./data/best_model/final_model \
    --episodes 20 \
    --results-dir ./data/results/

# 使用随机策略
python -m src.evaluation.evaluate ./data/best_model/final_model --stochastic
```

### 4. 可视化结果

```bash
# 生成轨迹图
python -m src.evaluation.trajectory_vis \
    --results-dir ./data/results/ \
    --output-dir ./data/results/visualizations/

# 使用 TensorBoard 查看
tensorboard --logdir ./data/logs/
```

## 配置

编辑 `src/training/config.yaml` 可自定义：

- **算法**：PPO（推荐）或 DQN
- **学习率**：PPO 为 3e-4，DQN 为 1e-4
- **N Steps**：2048（轨迹采样长度）
- **目标点**：航点坐标
- **奖励塑形**：距离系数、碰撞惩罚、成功奖励

示例：
```yaml
ppo:
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  gamma: 0.99

training:
  total_timesteps: 1000000
  checkpoint_freq: 10000
```

## 训练

### 典型训练流程

1. **初始化**（秒级）
   - 连接 AirSim
   - 验证环境
   - 初始化模型

2. **早期学习**（分钟级）
   - 模型探索环境
   - 出现首次碰撞
   - 奖励不再完全随机

3. **收敛阶段**（小时级）
   - 成功率提升
   - 每回合奖励趋于稳定
   - 策略进入平台期

### 监控训练进度

训练期间可使用 TensorBoard：
```bash
tensorboard --logdir ./data/logs/
```

关键指标：
- `rollout/ep_rew_mean`：平均回合奖励
- `train/policy_loss`：策略梯度损失
- `train/value_loss`：价值函数损失

### 停止条件

- 达到奖励阈值（可配置）
- 连续 N 个回合无提升（耐心值）
- 达到最大训练步数（默认 1M）

## 评估

### 单模型评估

```python
from src.evaluation.evaluate import evaluate_model

results = evaluate_model(
    model_path="./data/best_model/final_model",
    n_episodes=20,
    deterministic=True
)
```

### 多模型对比

```python
from src.evaluation.evaluate import compare_models

comparison = compare_models(
    model_paths=[
        "./data/checkpoints/drone_model_100000_steps.zip",
        "./data/checkpoints/drone_model_500000_steps.zip"
    ],
    n_episodes=10
)
```

### 关键指标

- **成功率**：无碰撞到达目标的回合占比
- **碰撞率**：发生碰撞的回合占比
- **平均奖励**：每回合平均奖励
- **路径长度**：累计欧氏距离
- **回合长度**：完成任务所需步数

## 可视化

### 轨迹图

```bash
python -m src.evaluation.trajectory_vis --results-dir ./data/results/
```

会生成：
- 每个回合的 3D 轨迹
- 高度变化曲线
- 与原点距离随时间变化曲线

### TensorBoard

```bash
tensorboard --logdir ./data/logs/
```

可查看：
- 奖励曲线
- 损失趋势
- 训练稳定性

## 结果

预期性能（完成训练后）：

| 指标 | 数值 |
|--------|-------|
| 成功率 | 70-90% |
| 碰撞率 | 5-20% |
| 平均奖励 | -20 到 +30 |
| 平均回合长度 | 100-200 步 |

## 故障排查

### 环境连接问题

**问题**：`Failed to connect to AirSim`

**解决方案**：
1. 确认 AirSim 模拟器已启动
2. 检查 `config.yaml` 中的 IP 地址（默认：127.0.0.1）
3. 确认 41451 端口可访问

### 强化学习训练问题

**问题**：奖励值保持不变

**解决方案**：
- 检查 `src/utils/reward_shaper.py` 中的奖励塑形逻辑
- 确认碰撞检测是否正常工作
- 检查 `data/logs/` 中是否有异常模式

**问题**：模型不收敛

**解决方案**：
- 适当提高学习率
- 降低 batch size
- 检查观测空间归一化

**问题**：CUDA/GPU 问题

**解决方案**：
```bash
# 强制使用 CPU 训练
python -m src.training.train --device cpu
```

## 高级用法

### 自定义奖励函数

编辑 `src/utils/reward_shaper.py`：

```python
class CustomRewardShaper(SimpleRewardShaper):
    def compute_reward(self, position, collision, reached_target, info):
        # 在这里实现你的自定义逻辑
        reward = ...
        return float(reward), done
```

### 多环境并行训练

在 `config.yaml` 中设置：
```yaml
hardware:
  n_envs: 4
  vec_env_type: "subproc"
```

### 使用不同算法

```bash
# 使用 DQN 训练
python -m src.training.train --algorithm DQN --timesteps 500000
```

## 后续改进

- [ ] 多智能体协同
- [ ] 动态障碍物
- [ ] Sim-to-real 迁移
- [ ] 用于部分可观测场景的 LSTM
- [ ] 课程学习
- [ ] 安全约束强化学习

## 参考资料

- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [AirSim Documentation](https://microsoft.github.io/AirSim/)
- [Gymnasium](https://gymnasium.farama.org/)

## 许可证

MIT

## 联系方式

如有问题或建议，请在仓库中提交 issue。
