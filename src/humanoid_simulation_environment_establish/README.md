# 人形机器人仿真环境（基于MuJoCo）

## 项目概述

本项目实现了一个基于MuJoCo物理引擎的人形机器人仿真环境，支持带空隙的走廊导航任务。项目包含交互式键盘控制、深度学习步态学习、无重力模式等多种功能，适用于机器人控制算法研究、步态优化和强化学习训练。

## 核心特性

- **交互式键盘控制**：支持实时键盘输入控制机器人前进、后退、转向
- **深度学习控制器**：使用神经网络（MLP + LSTM）学习最优步态和姿态控制
- **无重力模式**：支持无重力环境下的机器人运动仿真
- **带空隙走廊环境**：构建包含平台和空隙的复杂走廊环境
- **实时可视化**：使用MuJoCo viewer进行实时3D可视化

## 环境准备

### 依赖库安装

```bash
pip install mujoco numpy
```

- `mujoco`：MuJoCo物理引擎（版本 >= 2.3.0）
- `numpy`：数值计算支持

### 模型文件

确保项目目录下存在 `humanoid.xml` 文件，该文件定义了人形机器人的结构、关节和执行器。

## 项目结构

| 文件名 | 功能描述 |
|--------|----------|
| `main.py` | 主程序文件，包含环境、控制器和主循环 |
| `humanoid.xml` | 人形机器人模型文件（MJCF格式） |
| `README.md` | 项目说明文档 |

## 核心类说明

### 1. GapCorridorEnvironment（带空隙走廊环境）

基于MuJoCo的仿真环境，支持带空隙的走廊场景。

**主要功能：**
- 构建包含平台和空隙的走廊环境
- 支持有重力/无重力两种模式
- 提供观测、奖励和终止条件
- 无重力模式下的姿态稳定控制

**关键参数：**
- `corridor_length`：走廊总长度（默认100米）
- `corridor_width`：走廊宽度（默认10米）
- `use_gravity`：是否启用重力（默认False，无重力模式）

### 2. KeyboardController（键盘控制器）

交互式键盘控制器，支持实时控制机器人运动。

**控制指令：**
- `w` / `↑`：前进
- `s` / `↓`：后退
- `a` / `←`：左转
- `d` / `→`：右转
- `空格`：暂停/继续
- `r`：重置环境
- `q`：退出程序

**核心特性：**
- 自然的人类步态生成（支撑相和摆动相协调）
- 动作平滑处理（低通滤波 + 神经网络平滑）
- PID速度控制
- 与深度学习控制器混合使用

### 3. DeepLearningController（深度学习控制器）

使用神经网络学习最优步态和姿态控制策略。

**网络结构：**
- **策略网络**：MLP（2层） + LSTM + 输出层
  - 输入：状态 + 步态相位编码 + 上次动作
  - 输出：动作（执行器控制信号）
- **价值网络**：MLP（3层）
  - 输入：状态
  - 输出：状态价值估计

**学习机制：**
- 经验回放缓冲区（最大10000条经验）
- 策略梯度更新（REINFORCE算法）
- 步态相位编码（sin/cos编码）

## 使用方法

### 基本运行

```bash
python main.py
```

程序启动后会：
1. 初始化带空隙的走廊环境（无重力模式）
2. 启动MuJoCo交互式查看器
3. 等待键盘输入控制机器人运动

### 交互操作

1. **在查看器窗口内按键盘**（窗口需要有焦点）
2. 使用 `w/a/s/d` 或方向键控制机器人移动
3. 按 `空格` 暂停/继续仿真
4. 按 `r` 重置环境
5. 按 `q` 或关闭窗口退出程序

### 代码示例

```python
# 创建环境（无重力模式）
env = GapCorridorEnvironment(
    corridor_length=100, 
    corridor_width=10, 
    use_gravity=False
)

# 创建键盘控制器
controller = KeyboardController(
    env.model.nu, 
    env.get_actuator_indices()
)

# 主循环
while viewer_handle.is_running():
    action = controller.get_action(
        dt=env.control_timestep,
        current_velocity=current_vel,
        state=obs,
        reward=total_reward
    )
    obs, reward, done = env.step(action)
```

## 步态生成原理

### 人类步态特点

- **支撑相**：约占60%的时间，腿伸直、踝关节跖屈、推进身体
- **摆动相**：约占40%的时间，抬腿、膝关节弯曲、踝关节背屈

### 步态协调

- 左右腿相位相差180度，形成交替步态
- 髋关节前后摆动产生主要推进力
- 膝关节和踝关节配合髋关节动作
- 躯干轻微摆动保持平衡

## 无重力模式特性

在无重力模式下，系统实现了以下约束和控制：

1. **位置约束**：
   - 固定Z高度（防止飘起）
   - 固定Y位置（保持在走廊中心）
   - 允许X方向自由移动

2. **姿态稳定**：
   - 保持身体直立（roll和pitch为0）
   - 允许绕Z轴旋转（yaw，用于转向）
   - 头部高度稳定控制

3. **速度控制**：
   - 根据髋关节摆动计算前进速度
   - 根据髋关节外展和躯干旋转计算转向角速度
   - PID控制器平滑速度过渡

## 深度学习控制器配置

### 启用/禁用深度学习

在 `KeyboardController` 初始化时设置：

```python
controller.use_deep_learning = True  # 启用深度学习控制器
```

### 混合比例

传统动作和深度学习动作的混合比例（在 `get_action` 方法中）：

```python
raw_action = 0.7 * traditional_action + 0.3 * dl_action
```

### 模型保存/加载

```python
# 保存模型
controller.deep_controller.save_model("model.pkl")

# 加载模型
controller.deep_controller.load_model("model.pkl")
```

## 参数调整指南

### 步态参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `step_frequency` | 0.9 Hz | 步频，控制步行速度 |
| `step_duration` | 0.5 秒 | 每次按键的移动持续时间 |

### 动作平滑参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `action_smoothing_factor` | 0.7 | 动作平滑系数（0-1，越大越平滑） |

### 速度控制参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `_forward_velocity_gain` | 2.5 | 前进速度增益 |
| `_turn_velocity_gain` | 0.5 | 转向速度增益 |
| `_max_xy_velocity` | 2.0 m/s | 最大XY速度 |

### 深度学习参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `learning_rate` | 0.001 | 学习率 |
| `batch_size` | 64 | 批次大小 |
| `update_frequency` | 10 | 每N步更新一次网络 |

## 输出信息

程序运行时会输出以下信息：

- 每100步输出一次身体部位位置（躯干、头部、骨盆、脚部、手部）
- 累计奖励值
- 键盘控制状态变化
- Episode结束时的总奖励

## 注意事项

1. **窗口焦点**：键盘输入需要在MuJoCo查看器窗口有焦点时才能生效
2. **模型文件**：确保 `humanoid.xml` 文件存在于项目目录
3. **执行器映射**：程序会自动从模型文件中提取执行器名称到索引的映射
4. **无重力模式**：当前默认使用无重力模式，如需启用重力，修改 `main()` 函数中的 `use_gravity=True`

## 技术细节

### 步态相位编码

使用 sin/cos 编码将步态相位（0-2π）编码为2维向量，使神经网络能够学习周期性步态模式。

### LSTM状态管理

策略网络使用简化的LSTM结构，保留隐藏状态和细胞状态，用于学习时序依赖关系。

### 经验回放

使用双端队列（deque）实现经验回放缓冲区，最大容量10000条经验，支持随机采样进行批量训练。

## 参考资料

- [MuJoCo官方文档](https://mujoco.readthedocs.io/)
- [MuJoCo Python API](https://mujoco.readthedocs.io/en/latest/python.html)
- [MuJoCo Viewer使用指南](https://mujoco.readthedocs.io/en/latest/programming.html#viewer)

## 许可证

本项目基于MuJoCo物理引擎开发，请遵循MuJoCo的许可证要求。
