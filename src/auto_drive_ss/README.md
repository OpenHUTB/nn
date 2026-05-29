# 基于SSL-PPO的CARLA自动驾驶强化学习项目
## 项目简介
本项目基于CARLA仿真平台，结合自监督学习SSL与PPO强化学习算法，实现自动驾驶车辆训练与仿真。

## 项目结构
- src/: 核心算法、训练、模型源码
- docs/: 实验截图与资料
- run_train.py: 项目一键运行入口
- install.md: 环境配置指南
- training_log.txt: 训练过程日志

## 运行方式
1. 启动 CARLA 模拟器
2. 安装依赖：pip install -r requirements.txt
3. 启动训练：python run_train.py

## 可视化
使用 TensorBoard 查看奖励、损失等训练曲线。