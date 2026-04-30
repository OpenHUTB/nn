# 🚗 RL-Based-Adaptive-Cruise-Control

本项目旨在利用**深度强化学习**（Deep Reinforcement Learning, DRL）技术，特别是 **PPO (Proximal Policy Optimization)** 算法，来训练一个智能体，实现**自适应巡航控制**（Adaptive Cruise Control, ACC）功能。

传统的 ACC 系统通常基于 PID 或 MPC（模型预测控制），依赖于精确的物理模型和繁琐的参数调优。本项目探索如何通过端到端的强化学习方法，让智能体在与环境的交互中自主学习最优的跟车策略，以实现安全、舒适且高效的自动驾驶体验。

---

## ✨ 功能特点

- 🤖 **强化学习驱动**：使用 `Stable-Baselines3` 库实现 PPO 算法，处理连续动作空间。
- 🚦 **自定义仿真环境**：基于 `Gym` 框架构建了简化的车辆跟驰环境，模拟前车随机加减速场景。
- ⚖️ **多目标优化**：奖励函数综合考虑了**速度跟踪**（保持设定速度）、**安全距离**（防止碰撞）和**乘坐舒适度**（避免急加减速）。
- 📊 **可视化支持**：包含简单的状态输出，方便调试和观察训练过程。

---

## 🛠️ 环境依赖

确保您的系统已安装 Python 3.7 或更高版本。

本项目主要依赖以下库：

- `gym` (>=0.21.0): 用于构建强化学习环境接口。
- `stable-baselines3` (>=1.6.0): 提供高性能的强化学习算法实现。
- `numpy`: 用于数值计算。
- `shimmy`: 用于兼容旧版 Gym 接口。

### 安装步骤

1. 克隆本项目到本地：
   ```bash
   git clone https://github.com/YOUR_USERNAME/RL-Based-Adaptive-Cruise-Control.git
   cd RL-Based-Adaptive-Cruise-Control
