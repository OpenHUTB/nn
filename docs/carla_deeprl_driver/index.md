# CARLA 深度强化学习自动驾驶系统

基于 CARLA Simulator 的强化学习自动驾驶项目 (v1.0.0)

## 📋 项目简介

本项目使用 **CARLA Simulator** 实现了一个基于深度强化学习的自动驾驶控制系统。核心目标是训练智能代理在复杂城市环境中自主导航，实现端到端的自动驾驶控制。

## 🚀 快速开始

### 运行演示

```bash
python demo.py
```

### 运行 A2C 算法

```bash
python main.py
```

### 运行 SAC 算法

```bash
python run_sac.py
```

## 🔧 核心功能

- **多视角显示**：第三人称跟随镜头 + 俯视视角小窗口
- **实时 HUD**：速度显示、档位显示 (D/R/N)
- **平滑镜头跟随**：流畅的相机追踪效果
- **碰撞检测**：实时碰撞警告

## 📁 项目结构

```
carla_deeprl_driver/
├── source/                      # 核心源代码
│   ├── agent.py                 # ActorCar 类
│   ├── carlaenv.py              # CARLA 环境封装
│   ├── model.py                 # A2C 模型
│   ├── sac.py                   # SAC 实现
│   └── trainer.py               # 训练循环
├── config.yaml                  # 配置文件
├── demo.py                      # 演示脚本
├── main.py                      # A2C 入口
├── run_sac.py                   # SAC 入口
└── README.md                    # 项目说明
```

## ⚙️ 配置说明

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `host` | CARLA server 地址 | localhost |
| `port` | CARLA server 端口 | 2000 |
| `car_num` | NPC 车辆数量 | 10 |
| `lr` | 学习率 | 0.001 |
| `gamma` | 折扣因子 | 0.99 |

## 📐 支持的 RL 算法

### A2C (Advantage Actor-Critic)
- 离散动作空间
- ResNet50 骨干网络
- 支持 4 种动作：直行、左转、右转、刹车

### SAC (Soft Actor-Critic)
- 连续动作空间
- 自动熵调整
- 转向控制范围：[-1, 1]

## 🛠️ 环境要求

- **CARLA Simulator**: 0.9.13
- **Python**: 3.6+
- **PyTorch**: 1.8+
- **OpenCV**: 4.5+