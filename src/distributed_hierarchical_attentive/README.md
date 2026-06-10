# 分布式分层注意力网络 (Distributed Hierarchical Attentive Network)

基于 IMPALA 架构的分布式自动驾驶决策模型训练系统。

## 项目简介

IMPALA 架构协调多个工作者（使用 P-DQN 策略在 CARLA 中收集数据）和一个学习者（使用 V-trace 校正的 P-DQN 算法从共享队列中学习），实现了自动驾驶决策模型的高效、稳定分布式训练。

## 环境要求

- Python 3.7+
- CARLA 模拟器
- PyTorch
- Ray/分布式框架

## 快速开始

```bash
python main.py
```
