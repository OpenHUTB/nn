# 自动驾驶导航系统（DQN）

基于深度学习与 DQN 强化学习的 CARLA 自动驾驶导航系统。

---

<table>
  <tr>
    <td><a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.7-blue.svg" alt="Python" width="100%"></a></td>
    <td><a href="https://carla.org/"><img src="https://img.shields.io/badge/CARLA-0.9.13-orange.svg" alt="CARLA" width="100%"></a></td>
    <td><a href="https://www.tensorflow.org/"><img src="https://img.shields.io/badge/TensorFlow-2.5.0-yellow.svg" alt="TensorFlow" width="100%"></a></td>
    <td><a href="https://ubuntu.com/"><img src="https://img.shields.io/badge/Ubuntu-20.04-purple.svg" alt="Ubuntu" width="100%"></a></td>
  </tr>
</table>

---

## 📑 目录

- [项目简介](#项目简介)
- [核心功能](#核心功能)
- [系统架构](#系统架构)
- [模块说明](#模块说明)
- [状态空间与动作空间](#状态空间与动作空间)
- [奖励函数](#奖励函数)
- [DQN 模型架构](#dqn-模型架构)
- [配置系统](#配置系统)
- [数据收集与可视化](#数据收集与可视化)
- [环境要求](#环境要求)
- [部署流程](#部署流程)
- [运行方法](#运行方法)
- [注意事项](#注意事项)
- [参考](#参考)

---

## 项目简介

`automatic_drive_deep_learning` 是一个在 [CARLA](https://carla.org/) 仿真环境中运行的自动驾驶系统，采用**双模型 DQN（Deep Q-Network）**策略，将刹车决策与驾驶转向决策解耦，分别由独立的神经网络负责。

系统利用深度摄像头（Depth Camera）与语义分割摄像头（Semantic Segmentation Camera）作为感知来源，实时获取前方障碍物距离与道路环境信息，结合全局路径规划器生成预定轨迹，驱动 Tesla Model 3 虚拟车辆从指定起点自主导航至终点。

项目同时提供 **ROS 版本**（位于 `carla_ros_ws/`），可通过 `roslaunch` 在 ROS Noetic 中部署与运行。

---

## 核心功能

1. **双模型协同决策**：刹车模型（Braking DQN）负责紧急制动判断，驾驶模型（Driving DQN）负责方向与速度控制，两者串联执行。
2. **多传感器融合**：同时挂载深度摄像头和语义分割摄像头，障碍物距离由深度图计算，车辆/行人目标由语义分割图像精确识别。
3. **全局路径规划**：使用 CARLA 内置 `GlobalRoutePlanner`，以 0.5m 采样精度在地图上生成从起点到终点的离散路径点序列。
4. **实时路线可视化**：在 CARLA 世界中渲染规划路线（红色）、历史轨迹（蓝色）及车辆方向箭头（绿色）。
5. **自适应视角跟随**：独立线程以 60Hz 平滑跟随车辆，支持自适应平滑系数，避免抖动。
6. **可选交通流仿真**：支持在场景中批量生成 NPC 车辆与行人，模拟真实交通环境。
7. **数据收集与图表生成**：逐步记录障碍物距离、横向偏差、角度差、速度、帧率和奖励值，Episode 结束后自动生成 CSV 和性能图表。

---

## 系统架构

```
automatic_drive_deep_learning/
├── main/                      # 推理主程序（加载预训练模型运行）
│   ├── main.py                # 主入口：初始化环境、加载模型、运行 Episode
│   ├── car_env.py             # CARLA 车辆环境：传感器挂载、状态计算、奖励与终止判断
│   ├── config.py              # 全局配置：轨迹、模型路径、动作、可视化、传感器等
│   ├── config_manager.py      # 高级配置管理（动态参数校验与加载）
│   ├── data_collector.py      # 数据收集与 Matplotlib 图表生成
│   ├── route_visualizer.py    # CARLA 世界中路线与标记的实时绘制
│   ├── vehicle_tracker.py     # 车辆位姿跟踪与视角平滑跟随线程
│   ├── traffic_manager.py     # NPC 车辆与行人的批量生成与管理
│   ├── trajectory_manager.py  # 轨迹路径点管理工具
│   ├── model_manager.py       # 模型加载与推理封装
│   └── get_location.py        # 获取 CARLA 中车辆当前坐标（用于配置轨迹）
├── test/                      # DQN 训练脚本
│   ├── braking_dqn.py         # 刹车模型训练（2 状态 → 2 动作）
│   ├── driving_dqn.py         # 驾驶模型训练（2 状态 → 5 动作）
│   ├── test_braking.py        # 刹车模型推理测试
│   ├── test_driving.py        # 驾驶模型推理测试
│   ├── pedestrians_1.py       # 行人场景测试 1
│   └── pedestrians_2.py       # 行人场景测试 2
├── carla_ros_ws/              # ROS 工作空间（可选）
│   └── src/
│       ├── carla_autonomous/  # ROS 包：launch 文件、节点脚本、预训练模型
│       └── CMakeLists.txt
└── requirements.txt           # Python 依赖列表
```

**三层分离设计**：

| 层级 | 文件 | 职责 |
|:-----|:-----|:-----|
| **决策层** | `main.py`, `model_manager.py` | 加载双模型，串联推理，分发动作 |
| **环境层** | `car_env.py` | 传感器数据处理，状态提取，奖励计算，Episode 管理 |
| **仿真层** | CARLA Server | 物理仿真，Actor 管理，地图渲染 |

---

## 模块说明

### `car_env.py` — 车辆环境

核心环境类 `CarEnv`，实现：

- **`__init__(start, end)`**：连接 CARLA（localhost:2000），初始化车辆蓝图与目标终点。
- **`reset()`**：生成车辆（Tesla Model 3），挂载深度摄像头、语义分割摄像头、碰撞传感器、车道入侵传感器，调用 `GlobalRoutePlanner` 生成路径，返回初始状态。
- **`step(action, current_state)`**：执行 6 选 1 动作控制，处理图像，计算横向偏差与角度差，判断终止条件，返回 `(new_state, reward, done, waypoint)`。
- **`process_images()`**：从深度图计算到最近车辆/行人的距离，从语义分割图提取障碍物像素索引。

### `main.py` — 主程序入口

- `setup_environment()`：初始化 CARLA 环境、交通管理器、路线可视化器与车辆跟踪器。
- `load_models()`：加载刹车模型 `Braking___282.model` 和驾驶模型 `Driving__6030.model`。
- `predict_action()`：先由刹车模型决定是否制动（action=0）；若允许行驶，再由驾驶模型选择方向动作（action=1~5）。
- `run_episode()`：主循环，每步调用视角更新、状态获取、动作推理、环境执行与数据记录。

### `config.py` — 配置文件

集中管理所有参数，分为以下模块：

| 配置段 | 内容 |
|:-------|:-----|
| `TRAJECTORIES` | 预定义轨迹（起点/终点坐标） |
| `MODEL_PATHS` | 刹车模型与驾驶模型文件路径 |
| `ACTION_NAMES` | 动作名称映射（中文） |
| 交通配置 | NPC 车辆数量、行人数量、安全模式开关 |
| 可视化配置 | 路线颜色、箭头长度、历史路径最大点数 |
| 视角配置 | 跟随高度、俯视角、平滑系数、自适应开关 |
| 传感器配置 | 摄像头分辨率（640×480）、FOV（40°） |
| 数据收集配置 | 保存目录、图表 DPI、采样率 |

---

## 状态空间与动作空间

### 状态空间（4 维向量）

主程序推理阶段使用完整的 4 维状态：

| 索引 | 含义 | 归一化方式 |
|:----:|:-----|:----------|
| `s[0]` | 到前方障碍物的距离 | `(distance - 300) / 300` |
| `s[1]` | 车辆当前速度 | `(kmh - 30) / 30 - s[0]` |
| `s[2]` | 角度差 φ（车辆朝向 vs 路径朝向） | 原始度数 |
| `s[3]` | 横向偏差（到路径中心线的有符号距离） | `signed_dis × 15` |

> **刹车模型**仅使用 `s[0:2]`（障碍物距离、速度）。  
> **驾驶模型**仅使用 `s[2:4]`（角度差、横向偏差）。

### 动作空间（6 离散动作）

| 动作 ID | 名称 | 控制参数 |
|:-------:|:-----|:---------|
| **0** | 刹车 | `throttle=0, brake=1.0` |
| **1** | 直行 | `throttle=0.4, steer=0.0` |
| **2** | 左转 | `throttle=0.1, steer=-0.3` |
| **3** | 右转 | `throttle=0.1, steer=+0.3` |
| **4** | 微左 | `throttle=0.3, steer=-0.1` |
| **5** | 微右 | `throttle=0.3, steer=+0.1` |

---

## 奖励函数

### 刹车模型奖励

| 条件 | 动作 | 奖励 |
|:-----|:-----|:-----|
| 障碍物距离 < 制动安全距离 | 刹车（0） | `+3` |
| 障碍物距离 < 制动安全距离 | 油门（1） | `-3` |
| 障碍物距离充足 | 油门（1） | `+2` |
| 已停车 + 障碍物距离 < 150m + kmh=0 | — | `+200`（成功） |
| 碰撞 | — | `-200`（失败） |

### 驾驶模型奖励

| 条件 | 奖励 |
|:-----|:-----|
| `|φ| < 5°`，执行直行 | `+2` |
| `5° ≤ |φ| < 10°`，执行对应微调 | `+1~+2` |
| `|φ| ≥ 10°`，执行对应大幅转向 | `+1~+2` |
| 横向偏差 `|d| < 0.1` 且执行直行 | `+4` |
| 速度 > 20 km/h | `+1` |
| 速度 < 5 km/h | `-1` |
| 横向偏差 `|d| > 2m` | `-10` |
| 碰撞 / 角度偏差 > 100° / 偏离道路 > 3m | `-200`（终止） |
| 到达终点 (dist < 5m) | `+100`（成功） |

---

## DQN 模型架构

### 刹车模型（`braking_dqn.py`）

```text
输入: [障碍物距离, 速度]  (shape: 2,)
    │
    └─ Dense(2, activation='linear')
    │
输出: Q 值 (shape: 2,)   # [刹车, 油门]
```

| 参数 | 值 |
|:-----|:---|
| 优化器 | Adam (lr=0.0001) |
| 损失函数 | MSE |
| 经验回放容量 | 5000 |
| Minibatch 大小 | 16 |
| 折扣因子 γ | 0.99 |
| ε 衰减 | 0.95（每 Episode） |
| 训练 Episodes | 40 |

### 驾驶模型（`driving_dqn.py`）

```text
输入: [角度差 φ, 横向偏差 d]  (shape: 2,)
    │
    ├─ Dense(8, activation='relu')
    │
    └─ Dense(5, activation='linear')
    │
输出: Q 值 (shape: 5,)   # [直行, 左转, 右转, 微左, 微右]
```

| 参数 | 值 |
|:-----|:---|
| 优化器 | Adam (lr=0.0001) |
| 损失函数 | MSE |
| 经验回放容量 | 5000 |
| Minibatch 大小 | 16 |
| 折扣因子 γ | 0.99 |
| ε 衰减 | 0.95（每 Episode） |
| 训练 Episodes | 40 |
| 目标网络更新频率 | 每 2 步 |

---

## 配置系统

### 轨迹配置

```python
TRAJECTORIES = {
    "test_trajectory": {
        "start": [98.26, 99.16, 0.49, 0],   # [x, y, z, yaw]
        "end":   [74.65, -56.01, 0.41],      # [x, y, z]
        "description": "测试轨迹 - 直线道路"
    },
    "custom_trajectory": {
        "start": [108.92, 101.36, 0.61, 0],
        "end":   [62.21, -68.13, 0.74],
        "description": "自定义轨迹 - 城镇道路"
    }
}
CURRENT_TRAJECTORY = "test_trajectory"   # 切换此值选择轨迹
```

### 训练配置

```python
TOTAL_EPISODES = 3           # 运行的总 Episode 数
MAX_STEPS_PER_EPISODE = 2000 # 每个 Episode 最大步数
EPISODE_INTERVAL = 2.0       # Episode 间隔（秒）
```

### 交通配置

```python
ENABLE_TRAFFIC = False   # 是否生成 NPC 交通
TRAFFIC_VEHICLES = 30    # NPC 车辆数量
TRAFFIC_WALKERS = 50     # NPC 行人数量
TRAFFIC_SAFE_MODE = True # 仅生成安全车型
```

### 视角配置

```python
FOLLOW_DISTANCE = 8.0         # 跟随距离（米）
FOLLOW_HEIGHT = 3.0           # 跟随高度（米）
FOLLOW_PITCH = -20.0          # 俯视角（度）
VIEW_UPDATE_FPS = 60          # 视角更新频率（Hz）
VIEW_UPDATE_THREADED = True   # 启用独立线程
SMOOTH_FOLLOW_FACTOR = 0.01   # 平滑系数（越小越平滑）
```

---

## 数据收集与可视化

系统在每个 Episode 结束后自动生成以下输出：

**CSV 文件**（`data_logs/episode_data_<timestamp>.csv`）：

| 字段 | 含义 |
|:-----|:-----|
| 时间戳 | 相对运行时间（秒） |
| 障碍物距离 | 前方最近车辆/行人距离（米） |
| 动作编号/名称 | 执行的动作 ID 与中文名 |
| 横向偏差 | 到路径中心线的有符号偏移（米） |
| 角度差 | 车辆朝向与路径方向的夹角（度） |
| 速度_kmh | 车辆速度（km/h） |
| 帧率 | 推理帧率（FPS） |
| 奖励值 | 当步奖励 |

**可视化图表**（`data_logs/plots/`）：

- 障碍物距离与动作分布图
- 横向偏差与角度差时序图
- 速度与帧率时序图
- 六合一综合性能图表

---

## 环境要求

| 组件 | 版本 |
|:-----|:-----|
| 操作系统 | Ubuntu 20.04 |
| Python | 3.7 |
| CARLA 模拟器 | 0.9.13 |
| ROS（可选） | Noetic |
| TensorFlow | 2.5.0 |
| CUDA（GPU 加速） | 推荐 11.x |

---

## 部署流程

### 第一步：获取 CARLA 模拟器

从 CARLA 官方 Release 页面下载 0.9.13 版本并解压：

```bash
# 下载 CARLA 0.9.13（Linux）
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.13.tar.gz
tar -xzf CARLA_0.9.13.tar.gz -C ~/carla-0.9.13
```

### 第二步：克隆项目

```bash
git clone https://github.com/zhang0323/nn.git
cd nn
```

### 第三步：创建 Python 环境并安装依赖

```bash
# 使用 conda 创建隔离环境（推荐）
conda create -n carla-env python=3.7
conda activate carla-env

# 安装依赖（使用国内镜像加速）
pip install -r src/automatic_drive_deep_learning/requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> **注意**：`requirements.txt` 中的 `carla==0.9.13` 需要与 CARLA 安装版本一致。  
> 如果 `pip` 找不到该版本，可手动从 CARLA 安装目录安装：
> ```bash
> pip install ~/carla-0.9.13/PythonAPI/carla/dist/carla-0.9.13-cp37-cp37m-linux_x86_64.whl
> ```

### 第四步（可选）：构建 ROS 工作空间

> **仅在需要 ROS 集成时执行此步骤。**

```bash
cd src/automatic_drive_deep_learning/carla_ros_ws

# 安装 ROS Noetic（若未安装）
# http://wiki.ros.org/noetic/Installation/Ubuntu

# 构建 catkin 工作空间
catkin_make
source devel/setup.bash

# 安装 carla_autonomous 包内的额外依赖
cd src/carla_autonomous/utils
./install.sh
```

### 第五步：配置模型路径与轨迹

编辑 `src/automatic_drive_deep_learning/main/config.py`：

```python
# 1. 确认模型路径正确（预训练模型位于 carla_ros_ws/src/carla_autonomous/models/）
MODEL_PATHS = {
    'braking': '/path/to/nn/src/automatic_drive_deep_learning/carla_ros_ws/src/carla_autonomous/models/Braking___282.model',
    'driving': '/path/to/nn/src/automatic_drive_deep_learning/carla_ros_ws/src/carla_autonomous/models/Driving__6030.model',
}

# 2. 选择使用的轨迹
CURRENT_TRAJECTORY = "test_trajectory"  # 或 "custom_trajectory"
```

如需获取新的轨迹坐标，先在 CARLA 中运行：

```bash
cd src/automatic_drive_deep_learning/main
python get_location.py   # 打印当前车辆的 (x, y, z) 坐标
```

---

## 运行方法

### 基础版本（无 ROS）

**终端 1 — 启动 CARLA 模拟器：**

```bash
cd ~/carla-0.9.13
./CarlaUE4.sh
# 或使用无渲染模式（提升训练性能）
./CarlaUE4.sh -RenderOffScreen
```

**终端 2 — 运行主推理程序：**

```bash
conda activate carla-env
cd src/automatic_drive_deep_learning/main
python main.py
```

**快速测试（单独测试刹车/驾驶模型）：**

```bash
# 测试刹车模型推理
python ../test/test_braking.py

# 测试驾驶模型推理
python ../test/test_driving.py
```

---

### 训练新模型

```bash
conda activate carla-env
cd src/automatic_drive_deep_learning/test

# 训练刹车模型（约 40 个 Episode）
python braking_dqn.py
# 模型将保存至 test/models/Braking__<reward>max_...<timestamp>.model

# 训练驾驶模型（约 40 个 Episode）
python driving_dqn.py
# 模型将保存至 test/models/Driving__<reward>max_...<timestamp>.model
```

训练完成后，将生成的 `.model` 文件复制到 `config.py` 的 `MODEL_PATHS` 路径中。

---

### ROS 版本

**终端 1 — 启动 CARLA：**

```bash
./CarlaUE4.sh
```

**终端 2 — 一键启动 ROS 节点（推荐）：**

```bash
cd src/automatic_drive_deep_learning/carla_ros_ws/src/carla_autonomous/utils
./run_carla.sh
```

**或手动分步启动：**

```bash
# 终端 2：启动 ROS 节点
cd src/automatic_drive_deep_learning/carla_ros_ws
source devel/setup.bash
roslaunch carla_autonomous carla_autonomous.launch

# 终端 3：启动控制客户端
cd src/automatic_drive_deep_learning/carla_ros_ws
source devel/setup.bash
python src/carla_autonomous/scripts/carla_control_client.py
```

---

## 注意事项

1. **CARLA 必须先于主程序启动**，否则客户端连接（`localhost:2000`）会超时失败。
2. **模型文件路径**：`config.py` 中 `MODEL_PATHS` 使用绝对路径或相对于运行目录的路径，请根据实际情况修改。
3. **ROS 使用前必须 source**：每次新开终端都需执行 `source devel/setup.bash`，或将其加入 `~/.bashrc`。
4. **GPU 加速**：TensorFlow 默认使用 GPU（若可用），脚本会自动开启内存增长模式；如无 GPU，将降速运行于 CPU。
5. **数据保存**：运行时若 `data_logs/` 目录不存在，程序会自动创建；训练脚本的 `logs/` 目录用于 TensorBoard，可通过 `tensorboard --logdir logs/` 查看训练曲线。
6. **图表中文乱码**：Linux 下需安装中文字体（如文泉驿），程序会自动检测：
   ```bash
   sudo apt-get install fonts-wqy-microhei
   ```

---

## 参考

- 源码目录：[src/automatic_drive_deep_learning](https://github.com/zhang0323/nn/tree/main/src/automatic_drive_deep_learning)
- 参考项目：[varunpratap222/Autonomous-Vehicle-Navigation-Using-Deep-Learning](https://github.com/varunpratap222/Autonomous-Vehicle-Navigation-Using-Deep-Learning)
- CARLA 官方文档：[https://carla.readthedocs.io/en/0.9.13/](https://carla.readthedocs.io/en/0.9.13/)
- CARLA Python API 参考：[https://carla.readthedocs.io/en/0.9.13/python_api/](https://carla.readthedocs.io/en/0.9.13/python_api/)

## License

MIT
