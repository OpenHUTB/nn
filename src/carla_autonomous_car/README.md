# CARLA Frenet轨迹规划 - 自动驾驶强化学习系统

## 项目概述

本项目实现了一个基于强化学习的Frenet坐标系轨迹规划系统，用于CARLA模拟器中的自动驾驶车辆。系统采用PPO2（近端策略优化）算法结合LSTM网络，学习复杂交通场景下的最优驾驶策略。

## 核心特性

- **Frenet坐标系规划**：在Frenet坐标系下进行平滑的车道保持和变道轨迹规划
- **强化学习算法**：PPO2算法配合LSTM网络处理序列决策问题
- **安全层设计**：内置碰撞避免和安全约束机制
- **多实验配置**：支持基础、改进和Lyapunov稳定性三种实验配置
- **CARLA集成**：与CARLA模拟器深度集成，提供真实仿真环境

## 系统架构

```
carla_autonomous_car/
├── main.py                           # 主程序入口
├── config.py                         # 配置管理模块
├── agents/reinforcement_learning/    # 强化学习算法（stable_baselines）
├── carla_gym/envs/                   # CARLA环境接口
├── configs/                          # 实验配置文件
└── logs/                             # 训练和测试日志
```

## 安装指南

### 环境要求

- Python 3.6+
- CARLA 0.9.x 模拟器
- TensorFlow 1.14

### 依赖安装

```bash
pip install -r requirements.txt
```

主要依赖：
- tensorflow==1.14
- gym
- numpy, scipy, pandas
- pyyaml
- opencv-python
- pygame

## 使用说明

### 训练模式

使用基础配置训练智能体：

```bash
python main.py --cfg_file configs/experiment_baseline.yaml --agent_id 1
```

使用改进配置训练（带变道塑形）：

```bash
python main.py --cfg_file configs/experiment_improved.yaml --agent_id 2
```

使用Lyapunov安全约束配置训练：

```bash
python main.py --cfg_file configs/experiment_lyapunov.yaml --agent_id 3
```

### 测试模式

测试已训练模型：

```bash
python main.py --test --agent_id 1 --test_model best_100000
```

### 查看可用配置

```bash
python main.py --list_cfgs
```

## 配置详解

### 基础配置 (`experiment_baseline.yaml`)
- 标准PPO2算法配合LSTM网络
- 无变道塑形奖励
- 基础安全层保护

### 改进配置 (`experiment_improved.yaml`)
- 增加简单变道塑形奖励
- 增强探索策略
- 优化收敛性能

### Lyapunov配置 (`experiment_lyapunov.yaml`)
- 基于Lyapunov稳定性的安全约束
- 形式化安全保障
- 高级奖励塑形

## 奖励函数结构

系统采用多组件奖励函数：

- **速度奖励**：鼓励维持目标速度
- **安全奖励**：惩罚不安全距离和碰撞
- **车道保持**：奖励保持在正确车道内
- **舒适性**：惩罚急加速和急刹车
- **效率奖励**：奖励沿轨迹前进
- **变道塑形**（改进/Lyapunov配置）：指导变道决策

## 环境规格

- **观察空间**：局部传感器数据、车辆状态、轨迹信息
- **动作空间**：连续转向和加速控制
- **仿真环境**：带交通车辆的CARLA驾驶环境
- **安全层**：实时碰撞避免和紧急制动

## 结果保存结构

训练结果保存在 `logs/agent_{ID}/` 目录：
- `models/` - 训练模型检查点
- `config.yaml` - 训练使用的配置文件
- `reproduction_info.txt` - 训练元数据和参数
- `training_summary.json` - 训练时长和统计信息

## 性能指标

- **平均奖励**：主要性能指标
- **成功率**：完成回合的百分比
- **变道性能**：安全与不安全变道统计
- **碰撞率**：碰撞发生频率
- **压线时间**：超出可行驶区域的时间

## 高级功能

### 安全层
- 碰撞时间（TTC）监控
- 最小安全距离强制
- 紧急干预能力

### Lyapunov稳定性
- 形式化稳定性保证
- 自适应安全边界
- 保守探索策略

### 多模态训练
- 支持不同网络架构（MLP, LSTM, CNN）
- 灵活的策略配置
- 自定义CNN特征提取器

## 故障排除

### 常见问题

1. **CARLA导入错误**：确保CARLA Python API已安装并在PYTHONPATH中
2. **依赖缺失**：运行 `pip install -r requirements.txt`
3. **配置错误**：检查YAML文件语法和参数名称
4. **内存问题**：减小批量大小或使用更小的网络

### 结构验证

运行结构测试验证安装：

```bash
python test_import.py
```

## 项目状态

**版本**：1.0.0（第一次提交）
**最后更新**：2026年4月
**状态**：完整且功能正常

本项目提供了一个使用强化学习进行自动驾驶轨迹规划的完整框架，系统已准备好在CARLA模拟器上进行训练和测试。

---

*注：本项目基于RL-frenet-trajectory-planning-in-CARLA项目开发，适用于学术研究和教学用途。*