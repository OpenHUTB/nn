# MuJoCo 人形机器人仿真

## 项目概述

本模块基于 **MuJoCo 3.x** 物理引擎实现人形机器人仿真控制。机器人从 **平躺关键帧（home keyframe）** 启动，通过 PD 反馈控制维持平躺姿态，**右肘关节**按固定周期执行循环抬手动作（上升 → 保持 → 下降）。

机器人模型共有 **21 个自由度（DOF）**，包含躯干、下肢、双臂等关节，模型文件为 `humanoid.xml`。

## 功能特性

| 特性 | 说明 |
|:---|:---|
| **模型加载与初始化** | 从 `humanoid.xml` 加载 MJCF 模型，重置至 keyframe 0（平躺姿态），清零关节速度 |
| **PD 反馈控制** | 对 15 个躯干/下肢关节施加 PD 力矩，跟踪平躺关键帧的参考位置 |
| **肘关节周期抬手** | 右肘关节按 6 秒周期执行分段线性力矩轨迹（上升 2s → 保持 2s → 下降 2s） |
| **控制信号后处理** | 一阶指数平滑（EMA）滤波 + 力矩限幅，保证仿真数值稳定性 |
| **数据采集** | 以 200 Hz 频率将全状态数据（时间、qpos、qvel）写入 CSV 文件 |
| **实时可视化** | MuJoCo 原生渲染窗口，支持旋转/缩放/平移交互 |

## 模型结构

### 关节列表

| 索引 | 关节名称 | 类型 | 运动轴 | 关节范围 | 所属部位 |
|:---:|:---|:---|:---|:---|:---|
| 0 | `root` | freejoint | — | — | 根节点（躯干） |
| 1 | `abdomen_z` | hinge | Z | -45° ~ 45° | 腹部 |
| 2 | `abdomen_y` | hinge | Y | -75° ~ 30° | 腹部 |
| 3 | `abdomen_x` | hinge | X | -35° ~ 35° | 腹部 |
| 4 | `hip_x_right` | hinge | X | -35° ~ 15° | 右髋 |
| 5 | `hip_z_right` | hinge | Z | -60° ~ 35° | 右髋 |
| 6 | `hip_y_right` | hinge | Y | -150° ~ 20° | 右髋 |
| 7 | `knee_right` | hinge | -Y | -160° ~ 2° | 右膝 |
| 8 | `ankle_y_right` | hinge | Y | -50° ~ 50° | 右踝 |
| 9 | `ankle_x_right` | hinge | X | -50° ~ 50° | 右踝 |
| 10 | `hip_x_left` | hinge | -X | -35° ~ 15° | 左髋 |
| 11 | `hip_z_left` | hinge | -Z | -60° ~ 35° | 左髋 |
| 12 | `hip_y_left` | hinge | Y | -150° ~ 20° | 左髋 |
| 13 | `knee_left` | hinge | -Y | -160° ~ 2° | 左膝 |
| 14 | `ankle_y_left` | hinge | Y | -50° ~ 50° | 左踝 |
| 15 | `ankle_x_left` | hinge | -X | -50° ~ 50° | 左踝 |
| 16 | `shoulder1_right` | hinge | (2,1,1) | -85° ~ 60° | 右肩 |
| 17 | `shoulder2_right` | hinge | (0,-1,1) | -85° ~ 60° | 右肩 |
| 18 | `elbow_right` | hinge | (0,-1,1) | -100° ~ 50° | 右肘 |
| 19 | `shoulder1_left` | hinge | (-2,1,-1) | -85° ~ 60° | 左肩 |
| 20 | `shoulder2_left` | hinge | (0,-1,-1) | -85° ~ 60° | 左肩 |
| 21 | `elbow_left` | hinge | (0,-1,-1) | -100° ~ 50° | 左肘 |

### 执行器列表

共 **21 个执行器（motor）**，与关节一一对应。执行器力矩范围：`[-1, 1]` N·m。

### 传感器

| 传感器 | 类型 | 说明 |
|:---|:---|:---|
| `torso_acc` | 加速度计 | 躯干加速度 |
| `torso_vel` | 速度计 | 躯干速度 |
| `foot_force` | 力传感器 | 右脚受力 |

## 控制架构

```
┌─────────────────────────────────────────────────────────┐
│                    MuJoCo 仿真主循环                      │
├─────────────────────────────────────────────────────────┤
│  1. 读取当前状态 qpos, qvel                              │
│                        ↓                                │
│  2. PD 控制器（15 个稳定关节）                            │
│     tau = kp * (q_des - q_cur) - kd * q_vel             │
│                        ↓                                │
│  3. 肘关节开环力矩（分段线性轨迹）                         │
│     上升 2s → 保持 2s → 下降 2s                          │
│                        ↓                                │
│  4. EMA 平滑 + 力矩限幅                                  │
│     u_smooth[t] = α·u_target + (1-α)·u_smooth[t-1]      │
│     clip(u, -1.0, 1.0)                                  │
│                        ↓                                │
│  5. 写入 data.ctrl → mj_step() → 数据记录                │
└─────────────────────────────────────────────────────────┘
```

## 稳定关节列表

PD 控制器覆盖以下 **15 个稳定关节**：

```
abdomen_z, abdomen_y, abdomen_x,
hip_x_right, hip_z_right, hip_y_right, knee_right, ankle_y_right, ankle_x_right,
hip_x_left, hip_z_left, hip_y_left, knee_left, ankle_y_left, ankle_x_left
```

## 数据输出

- **终端输出**：每 0.5 秒打印仿真时间、抬手阶段、肘关节力矩、机器人高度
- **CSV 文件**：`simulation_data.csv`，包含 `[time, qpos_0..nq-1, qvel_0..nv-1]`，以 200 Hz 采样
