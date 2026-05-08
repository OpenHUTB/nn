# 🚗 RL-ACC

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/yourusername/RL-Based-Adaptive-Cruise-Control.svg?style=social)](https://github.com/yourusername/RL-Based-Adaptive-Cruise-Control)

本项目旨在利用**深度强化学习**（Deep Reinforcement Learning, DRL）技术，特别是 **PPO (Proximal Policy Optimization)** 算法，来训练一个智能体，实现**自适应巡航控制**（Adaptive Cruise Control, ACC）功能。

传统的 ACC 系统通常基于 PID 或 MPC（模型预测控制），依赖于精确的物理模型和繁琐的参数调优。本项目探索如何通过端到端的强化学习方法，让智能体在与环境的交互中自主学习最优的跟车策略，以实现安全、舒适且高效的自动驾驶体验。

---

## ✨ 功能特点

- 🤖 **强化学习驱动**：使用 `Stable-Baselines3` 库实现 PPO 算法，处理连续动作空间
- 🚦 **自定义仿真环境**：基于 `Gymnasium` 框架构建了简化的车辆跟驰环境，模拟前车随机加减速场景
- 🚗 **CARLA 集成**：支持在 CARLA 自动驾驶仿真器中进行真实场景测试
- ⚖️ **多目标优化**：奖励函数综合考虑了**速度跟踪**、**安全距离**和**乘坐舒适度**
- 📊 **可视化支持**：包含详细的状态输出和结果图表生成，方便调试和观察训练过程
- 🔧 **灵活配置**：通过 `config.py` 轻松调整各项参数

---

## 🛠️ 环境依赖

确保您的系统已安装 Python 3.7 或更高版本。

### 核心依赖

| 库 | 版本要求 | 说明 |
|----|---------|------|
| `gymnasium` | >=0.28.0 | 强化学习环境接口 |
| `stable-baselines3` | >=2.0.0 | 强化学习算法实现 |
| `numpy` | >=1.21.0 | 数值计算 |
| `torch` | >=1.10.0 | PyTorch 深度学习框架 |
| `matplotlib` | >=3.5.0 | 数据可视化 |
| `tensorboard` | >=2.9.0 | 训练日志可视化 |
| `tqdm` | >=4.64.0 | 进度条 |
| `rich` | >=10.0.0 | 富文本输出 |

### 安装步骤

1. 克隆本项目到本地：
   ```bash
   git clone https://github.com/yourusername/RL-Based-Adaptive-Cruise-Control.git
   cd RL-Based-Adaptive-Cruise-Control
   ```

2. 安装所需依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. （可选）安装 CARLA 仿真器：
   - 参考 [CARLA 官方文档](https://carla.readthedocs.io/en/latest/start_quickstart/) 安装 CARLA
   - 安装 CARLA Python API：
   ```bash
   pip install carla
   ```

---

## 🚀 快速开始

### 训练模型

运行以下命令开始训练 ACC 智能体：

```bash
python train.py
```

训练过程中：
- 模型会定期保存到 `models/` 目录下
- 训练日志会保存到 `logs/` 目录，可通过 TensorBoard 查看
- 每 `SAVE_FREQUENCY` 步保存一个检查点模型

### 测试模型

使用训练好的模型进行测试：

```bash
python test.py --model models/best_model.zip
```

可选参数：
- `--episodes`: 测试回合数（默认：10）
- `--render`: 是否渲染测试过程（默认：True）
- `--save-plots`: 是否保存图表（默认：True）

### 可视化结果

测试完成后，结果图表会自动保存到 `test_results/` 目录：
- `rewards.png`: 各回合奖励分布
- `performance.png`: 速度跟踪、车间距离、加速度变化
- `statistics.png`: 测试性能统计

---

## 📁 项目结构

```
RL-Based-Adaptive-Cruise-Control/
├── acc_env/                      # 自适应巡航控制环境
│   ├── __init__.py
│   └── acc_env.py               # 环境定义（状态空间、动作空间、奖励函数）
├── utils/                        # 工具函数
│   ├── __init__.py
│   └── reward_functions.py      # 奖励函数定义
├── models/                       # 训练好的模型保存目录
├── config.py                     # 配置文件（车辆参数、训练参数等）
├── train.py                      # 训练脚本
├── test.py                       # 测试脚本
├── visualize.py                  # 可视化脚本
├── requirements.txt              # 依赖列表
├── carla_inference.py            # CARLA 仿真推理脚本
├── carla_final.py                # CARLA 完整测试脚本
└── README.md                     # 项目说明
```

---

## 📝 配置说明

主要配置参数位于 `config.py` 文件中：

### 车辆物理参数
- `TARGET_SPEED`: 目标巡航速度 (m/s)，默认 25.0
- `SAFETY_DISTANCE`: 安全距离 (m)，默认 15.0
- `MAX_ACCELERATION`: 最大加速度 (m/s²)，默认 2.0
- `MAX_DECELERATION`: 最大减速度 (m/s²)，默认 -3.0
- `DT`: 时间步长 (s)，默认 0.1

### 训练参数
- `TRAINING_TIMESTEPS`: 总训练步数，默认 1,000,000
- `BATCH_SIZE`: 批次大小，默认 64
- `GAMMA`: 折扣因子，默认 0.99
- `LEARNING_RATE`: 学习率，默认 3e-4
- `SAVE_FREQUENCY`: 模型保存频率，默认 100,000

---

## 🔧 环境设计

### 状态空间
| 维度 | 含义 | 范围 |
|------|------|------|
| 0 | 自车速度 | [0, 35] m/s |
| 1 | 前车速度 | [0, 70] m/s |
| 2 | 两车相对距离 | [0, 200] m |
| 3 | 相对速度 | [-10, 10] m/s |
| 4 | 目标速度 | [0, 35] m/s |

### 动作空间
- 连续动作，表示自车加速度指令
- 范围：[-3.0, 2.0] m/s²

### 奖励函数
奖励函数综合考虑以下三个方面：
1. **速度跟踪奖励**：鼓励保持目标速度
2. **安全距离奖励**：惩罚过近或过远的跟车距离
3. **舒适度奖励**：惩罚剧烈的加减速变化

---

## 📊 结果评估

训练完成后，可以通过以下指标评估模型性能：

| 指标 | 说明 |
|------|------|
| **平均奖励** | 衡量整体性能 |
| **成功率** | 成功完成回合的比例（无碰撞） |
| **碰撞次数** | 发生碰撞的回合数 |
| **速度跟踪误差** | 实际速度与目标速度的平均偏差 |
| **距离保持误差** | 实际距离与期望安全距离的平均偏差 |

---

## 🚗 CARLA 仿真

项目包含多个 CARLA 相关脚本：

| 脚本 | 说明 |
|------|------|
| `carla_inference.py` | CARLA 环境中加载模型进行推理 |
| `carla_final.py` | CARLA 完整测试脚本 |
| `carla_extended.py` | 扩展功能测试 |
| `carla_control_full.py` | 完整控制测试 |

运行 CARLA 测试：
```bash
# 首先启动 CARLA 服务器
# 然后运行推理脚本
python carla_inference.py --model models/best_model.zip
```

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进这个项目！

### 贡献步骤
1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

感谢以下项目和资源的启发：

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [OpenAI Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- [CARLA Simulator](https://github.com/carla-simulator/carla)
- [PyTorch](https://pytorch.org/)

---

## 📬 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 [Issue](https://github.com/yourusername/RL-Based-Adaptive-Cruise-Control/issues)
- 发送邮件至 your@email.com

---

*🚀 Happy Driving!*