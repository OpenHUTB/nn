# 人形机器人仿真环境（基于MuJoCo与dm_control）

## 项目概述
本项目构建了一系列人形机器人仿真环境，基于MuJoCo物理引擎和dm_control框架，支持机器人运动控制、走廊导航、目标追踪等多种任务。通过模块化设计，实现了机器人模型加载、环境配置、控制策略与可视化展示的完整流程，适用于机器人控制算法研究与验证。

## 环境准备

### 依赖库安装
```bash
pip install mujoco dm-control absl-py numpy
```
- `mujoco`：物理引擎核心，负责动力学计算
- `dm-control`：DeepMind开发的高级仿真框架，简化环境与任务构建
- `absl-py`：命令行参数处理工具
- `numpy`：数值计算支持

## 项目结构

| 文件名             | 功能描述                                                     |
|-------------------|--------------------------------------------------------------|
| `main.py`         | 核心控制程序，实现工作目录自动切换、模型加载与基础运动控制       |
| `main_humanoid.py`| 软体机器人弹跳仿真程序，基于比例控制实现从蹲姿到站姿的运动       |
| `run_walls.py`    | 带随机墙壁的走廊奔跑环境，使用dm_control构建强化学习任务         |
| `motion_main.py`  | 扩展运动环境，支持墙壁走廊和间隙走廊两种场景切换                 |
| `go_to_target.py` | 目标点追踪任务环境，实现机器人向指定目标位置移动的仿真           |
| `humanoid.xml`    | MJCF格式机器人模型文件，定义人形机器人的结构、关节与执行器       |
| `README.md`       | 项目说明文档                                                   |

## 核心功能

### 1. 基础运动控制（main.py & main_humanoid.py）
- 自动路径处理：程序启动时自动切换到项目根目录，确保相对路径正确
- 模型加载验证：检查并加载人形机器人模型，提供清晰的加载状态反馈
- 关键帧控制：基于预定义的"深蹲"和"站立"关键帧，通过比例控制实现平滑运动过渡
- 实时可视化：使用MuJoCo viewer展示机器人运动状态，输出时间与躯干高度等关键信息

```python
# 核心控制逻辑示例
qpos_error = target_qpos[7:] - data.qpos[7:]  # 计算关节位置误差
data.ctrl[:] = Kp * qpos_error  # 比例控制输出
```

### 2. 走廊奔跑环境（run_walls.py & motion_main.py）
- 动态场景生成：支持两种走廊类型（带墙壁/带间隙），墙壁宽度和间隙长度可随机化
- 任务定义：设定目标速度（3.0单位/秒），通过奖励函数引导机器人高效移动
- 多视角观测：支持第一视角相机和全局视角，便于观察与调试
- 参数可配置：走廊长度、宽度、墙壁高度等参数可灵活调整

```python
# 走廊环境配置示例
arena = corr_arenas.WallsCorridor(
    wall_gap=4.,  # 墙壁间隙宽度
    wall_width=distributions.Uniform(1, 7),  # 随机墙壁宽度
    wall_height=3.0,  # 墙壁高度
    corridor_length=100  # 走廊总长度
)
```

### 3. 目标追踪任务（go_to_target.py）
- 目标点导航：在平面环境中随机生成目标点，机器人需移动至目标位置
- 灵活的时间限制：单次仿真时长可配置（默认30秒）
- 兼容强化学习：环境接口符合强化学习标准，可直接用于训练策略

## 使用方法

1. 基础运动控制仿真：
```bash
python main.py
```

2. 走廊奔跑仿真（带墙壁）：
```bash
python run_walls.py
```

3. 间隙走廊仿真：
```bash
python motion_main.py
```

4. 目标追踪仿真：
```bash
python go_to_target.py
```

### 交互操作
- 仿真窗口支持视角旋转、缩放和平移
- 按`Ctrl+C`终止仿真程序

## 参数调整指南

| 参数               | 调整范围 | 效果说明                         |
|-------------------|----------|----------------------------------|
| `Kp`（比例增益）   | 1.0~10.0 | 增大会加快响应速度，过大会导致震荡 |
| `wall_gap`        | 2.0~6.0  | 减小值增加走廊穿过难度           |
| `target_velocity` | 1.0~5.0  | 提高值要求机器人运动速度更快     |
| `control_timestep`| 0.01~0.1 | 减小值提高控制精度（增加计算量） |

## 参考资料
- [MuJoCo官方文档](https://mujoco.readthedocs.io/)
- [dm_control框架文档](https://github.com/deepmind/dm_control)
- [CMU人形机器人模型说明](https://github.com/deepmind/dm_control/tree/main/dm_control/locomotion/walkers)
