# 基于 MuJoCo 的 ANYmal C 四足机器人仿真与控制

**仓库地址：** [https://github.com/Liu-Yirun/nn](https://github.com/Liu-Yirun/nn)

---

## 1. 项目概述

### 1.1 项目简介

本项目基于 MuJoCo 物理仿真引擎，加载 DeepMind MuJoCo Menagerie 中的 **ANYmal C** 四足机器人模型，实现模型加载、可视化仿真、周期性腿部摆动控制与运行状态监控。项目从 ANYmal B 模型起步，经路径适配、稳定性优化、工程化重构与功能增强，形成可跨设备运行的仿真基线。

### 1.2 项目整体流程

```
加载 MJCF 模型 → 配置初始位姿与物理参数 → 启动被动可视化窗口
      → 自动正弦摆动控制 → mj_step 物理步进 → 定时打印关节/速度日志
```

### 1.3 项目模块总览

| 模块 | 函数 | 核心功能 |
| ---- | ---- | -------- |
| 配置中心 | `CONFIG` | 集中管理模型路径、位姿、仿真步长、帧率、摆动参数 |
| 模型加载 | `load_mujoco_model` | 路径校验、异常捕获、模型信息打印 |
| 机器人初始化 | `configure_robot` | 设置基座位姿、重力、时间步、控制量清零 |
| 摆动控制 | `auto_swing_control` | 正弦周期驱动前 6 个关节，无需键盘输入 |
| 姿态复位 | `reset_robot` | 恢复基座、关节、速度与仿真时钟 |
| 仿真主循环 | `run_simulation` | 帧率控制、步进、画面同步、状态日志 |

### 1.4 核心技术特点

| 特点 | 说明 |
| ---- | ---- |
| 相对路径部署 | 使用 `anybotics_anymal_c/anymal_c.xml`，脱离固定盘符 |
| 模块化设计 | 配置、加载、控制、仿真循环分离，便于二次开发 |
| 自动摆动控制 | 正弦波驱动腿部关节，可开关、可调幅度与频率 |
| 运行监控 | 定时输出运行时长、仿真时间、关节角与机身线速度 |
| 稳定帧率 | 目标 60 FPS，避免仿真过快导致可视化卡顿 |

### 1.5 项目核心目标

1. 在 MuJoCo 中稳定加载并可视化 ANYmal C 四足机器人
2. 实现可配置的周期性腿部摆动，用于步态调试
3. 提供清晰的日志输出，支持仿真过程监控
4. 形成工程化代码结构，便于后续接入强化学习控制

---

## 2. 快速上手

### 2.1 运行环境

| 类别 | 要求 |
| ---- | ---- |
| 操作系统 | Windows 10/11 |
| Python | 3.8 及以上 |
| 物理仿真 | MuJoCo 2.2.2+ |

### 2.2 依赖安装

在项目目录（`mujoco_plex`）打开 PowerShell，执行：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2.3 启动仿真

```bash
python main.py
```

启动后将打开 MuJoCo 被动可视化窗口，机器人腿部自动周期摆动；关闭窗口即可退出。

### 2.4 常用配置参数

在 `main.py` 顶部 `CONFIG` 字典中修改：

```python
CONFIG = {
    "model_path": "anybotics_anymal_c/anymal_c.xml",
    "base_pos": (0.0, 0.0, 0.5),       # 基座初始位置
    "time_step": 0.002,                # 仿真步长
    "target_fps": 60,                    # 目标帧率
    "auto_swing_enable": True,         # 自动摆动开关
    "joint_swing_amp": 0.4,            # 摆动幅度
    "swing_freq": 1.0,                 # 摆动频率 (Hz)
    "print_interval": 1.0,             # 日志打印间隔 (秒)
}
```

---

## 3. 代码模块详解

### 3.1 模型加载 `load_mujoco_model`

- 校验路径类型与文件是否存在
- 调用 `mujoco.MjModel.from_xml_path` 加载 MJCF
- 打印机身、关节、执行器数量，便于调试索引

### 3.2 机器人初始化 `configure_robot`

- 设置 `base` 刚体初始位置与四元数姿态
- 配置仿真时间步与重力
- 将控制量 `data.ctrl` 清零

### 3.3 自动摆动 `auto_swing_control`

对前 6 个执行器施加正弦控制：

$$
u(t) = A \sin(2\pi f t)
$$

其中 $A$ 为 `joint_swing_amp`，$f$ 为 `swing_freq`。关闭 `auto_swing_enable` 时控制量归零。

### 3.4 仿真主循环 `run_simulation`

每帧执行：摆动控制 → `mj_step` → `viewer.sync` → 帧率休眠。按 `print_interval` 输出：

- 程序运行时长与 MuJoCo 仿真时钟
- 前 6 个关节角度（`qpos[7:13]`）
- 机身线速度（`cvel` 线速度分量）

---

## 4. 项目目录结构

```text
mujoco_plex/
├── main.py                 # 主程序入口
├── index.md                # 项目文档（本页）
├── mkdocs.yml              # MkDocs 站点配置
├── requirements.txt        # Python 依赖
├── anybotics_anymal_c/     # ANYmal C 模型资源
│   ├── anymal_c.xml
│   └── assets/             # 3D 网格 (.obj)
└── mjmodel.xml             # 备用模型文件
```

---

## 5. 开发迭代记录

### 提交 1：初始版本 — 本地绝对路径加载 ANYmal B

- 使用本地绝对路径加载 `anymal_b.xml`
- 实现基础 `mj_step` 与可视化窗口

### 提交 2：路径适配 — 相对路径部署

- 改为 `anybotics_anymal_b/anymal_b.xml`
- 支持将模型文件夹放在代码同级目录

### 提交 3：防掉落优化 — 基座固定与物理调参

- 锁定基座位姿，调整碰撞与摩擦参数
- 初始化关节控制量，保持站立姿态

### 提交 4：模型切换 — ANYmal C + 版权声明

- 切换为 `anybotics_anymal_c/anymal_c.xml`
- 补充 ANYbotics BSD 开源协议说明

### 提交 5：工程化重构 — CONFIG + 模块化

- 引入 `CONFIG` 配置字典
- 拆分加载、初始化、仿真循环函数
- 完善类型注解与异常处理

### 提交 6：稳定性修复 — qpos/qvel 实时约束

- 每帧重置根刚体位姿与速度，消除坠落与漂移

### 提交 7：高度微调

- 上调基座 Z 轴高度，避免与地面穿模

### 提交 8：功能增强 — 运行日志与关节监控

- 新增运行时长、仿真时间、关节角度定时打印
- 新增机身线速度输出

### 提交 9：自动摆动控制（当前版本）

- 移除键盘依赖，改为正弦周期自动驱动腿部
- 新增 `reset_robot` 复位函数
- 保留帧率控制与模块化结构

---

## 6. 优化总结

| 维度 | 改进内容 |
| ---- | -------- |
| 路径适配 | 绝对路径 → 相对路径，跨设备运行 |
| 稳定性 | 物理参数调优、位姿与速度约束，解决掉落与穿模 |
| 代码架构 | 配置中心化、函数模块化、完善异常处理 |
| 调试能力 | 关节角度、线速度、运行时长周期性日志 |
| 控制方式 | 自动正弦摆动，参数可在 CONFIG 中调节 |

---

## 7. 现存不足与后续方向

### 7.1 现存不足

- 当前为开环正弦摆动，未实现真实步态或平衡控制
- 未接入强化学习训练环境
- 缺少与地面的完整场景 XML（scene.xml）

### 7.2 后续优化

- 引入 PD 控制器实现支撑相 / 摆动相步态
- 搭建 Gymnasium 环境，接入 PPO / SAC 步态学习
- 增加域随机化与课程学习，提升策略鲁棒性
- 参考课程仓库中 `mujoco_running`、`mujoco_hci_sim` 项目的分层控制架构

---

## 8. 本地预览文档站点

安装依赖后，在项目目录执行：

```bash
mkdocs serve
```

浏览器访问 `http://127.0.0.1:8000` 预览本文档页面。

---

## 9. 总结

本项目完成了 ANYmal C 四足机器人在 MuJoCo 中的加载、可视化与周期性腿部控制，历经路径适配、稳定性修复与工程化重构，形成结构清晰、可扩展的仿真基线。代码支持参数化配置与运行监控，可作为四足机器人运动控制与强化学习研究的起点。
