<<<<<<< HEAD
# 人形机器人行走模拟

## 项目概述
本项目构建了基于 **ROS 1 (Noetic)** 与 **Mujoco** 仿真平台的人形机器人步态控制仿真系统，通过 CPG（中枢模式发生器）算法实现稳定步态生成，集成标准 ROS 接口与多模态控制方式，提供模块化、可扩展的代码库支持人形机器人运动控制算法研发。借助 Mujoco 的高保真物理仿真能力，模拟真实人形机器人的动力学特性与运动约束，为步态控制技术的学习、研究与开发提供可靠的仿真环境。

## 环境准备

### 依赖库安装
```bash
# 建议使用Python 3.8及以上版本（ROS Noetic默认Python 3.8）
pip3 install mujoco>=2.3.7 numpy
```

#### 依赖说明：
- mujoco>=2.3.7：高保真物理仿真核心库，提供机器人动力学计算与可视化支持
- numpy：数值计算基础库，支持数组运算与参数优化
- ros-noetic-geometry-msgs：ROS 标准几何消息类型（如速度指令 Twist）
- ros-noetic-sensor-msgs：ROS 标准传感器消息类型（如关节状态 JointState）
- ros-noetic-std-msgs：ROS 标准基础消息类型（如字符串 String）
- ros-noetic-rviz：ROS 可视化工具，支持机器人模型与状态实时展示

### 开发环境配置
1. 安装 Ubuntu 20.04 LTS 操作系统（ROS Noetic 适配版本）
2. 安装 ROS 1 Noetic [官方完整版](http://wiki.ros.org/noetic/Installation/Ubuntu)（推荐 `ros-noetic-desktop-full`）
3. 安装 [VSCode](https://code.visualstudio.com/) 并配置 Python 3.8+ 解释器
4. 推荐插件：Python、Pylance、ROS、Code Runner（提升 ROS 开发效率）

## 项目结构

| 文件名                  | 功能描述                                                     |
|-------------------------|--------------------------------------------------------------|
| `CMakeLists.txt`        | ROS 编译配置文件，声明依赖、安装规则与编译选项                 |
| `package.xml`           | ROS 包元信息文件，定义包名、版本、依赖与维护者信息             |
| `setup.py`              | Python 模块安装配置，确保脚本与模块可被 ROS 识别调用           |
| `launch/mujoco_manrun.launch` | ROS 一键启动文件，加载参数配置并启动仿真节点                   |
| `config/gait_params.yaml`    | 核心配置文件，存储 CPG 步态参数、仿真参数与关节控制参数         |
| `scripts/main.py`       | 仿真系统入口脚本，负责 ROS 节点初始化、模块加载与主循环控制     |
| `scripts/humanoid_stabilizer.py` | 核心控制模块，集成 CPG 步态生成、动力学解算与关节控制逻辑       |
| `scripts/utils.py`      | 通用工具函数库，包含数值裁剪、参数转换等辅助功能               |
| `models/humanoid.xml`   | Mujoco 机器人模型文件，定义人形机器人的连杆、关节与物理属性     |
| `README.md`             | 项目说明文档，包含环境配置、使用方法与常见问题解决             |

## 核心功能

### 1. 高保真动力学仿真
- 基于 Mujoco 实现人形机器人的精确动力学建模，还原关节摩擦、连杆质量等物理特性
- 支持 100Hz 高频率仿真，保证步态控制的实时性与稳定性
- 内置可视化窗口，实时展示机器人运动状态与关节轨迹

### 2. CPG 步态生成系统
- 基于中枢模式发生器（CPG）算法，支持 NORMAL/SLOW/FAST/原地踏步 四种步态模式
- 步态参数（频率、振幅）支持运行时动态调整，适配不同运动场景
- 内置关节角度裁剪与PD控制，确保运动安全与平滑过渡

### 3. 标准 ROS 接口集成
- 订阅 ROS 标准 `/cmd_vel` 话题，支持线速度/角速度指令输入
- 发布 `/joint_states` 话题，输出关节位置、速度与力矩数据
- 集成 ROS 参数服务器，支持仿真参数与步态参数实时配置

### 4. 多模态控制方式
- 键盘控制：支持 W/A/S/D 方向控制与数字键步态切换，操作直观
- ROS 话题控制：通过 `rostopic pub` 命令发布控制指令，便于算法集成
- 参数动态调整：通过 `rosparam set` 命令实时修改步态与仿真参数

## 使用方法

### 编译项目
```bash
# 进入 ROS 工作空间
cd ~/catkin_ws
# 编译功能包
catkin_make --cmake-args -DCMAKE_BUILD_TYPE=Release
# 激活 ROS 环境
source devel/setup.bash
```

### 一键启动仿真

```bash
# 启动 ROS 核心、仿真节点与可视化窗口
roslaunch mujoco_manrun mujoco_manrun.launch
```

### 手动分步启动（调试用）
```bash
# 1. 启动 ROS 核心（新终端）
roscore
# 2. 启动仿真节点（新终端，需先激活 ROS 环境）
rosrun mujoco_manrun main.py
# 3. 启动 RViz 可视化（可选，新终端）
rviz -d $(find mujoco_manrun)/config/rviz_config.rviz
```
### 控制指令示例
```bash
# 1. ROS 话题控制：发布前进指令（线速度 0.5 m/s）
rostopic pub -r 10 /cmd_vel geometry_msgs/Twist "{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
# 2. ROS 话题控制：发布左转指令（角速度 0.2 rad/s）
rostopic pub -r 10 /cmd_vel geometry_msgs/Twist "{linear: {x: 0.0}, angular: {z: 0.2}}"
# 3. 动态调整参数：修改 CPG 步态频率为 0.7 Hz
rosparam set /humanoid/cpg/freq 0.7
```

## 参数调整指南

| 参数路径                | 调整范围       | 效果说明                                                     |
|-------------------------|----------------|--------------------------------------------------------------|
| `/humanoid/cpg/freq`    | 0.3~1.0 Hz     | 调整 CPG 振荡器频率，数值越大步态越快（建议 0.5~0.7）          |
| `/humanoid/cpg/amp`     | 0.2~0.6        | 调整 CPG 振荡器振幅，数值越大关节摆动幅度越大（建议 0.4~0.5）  |
| `/humanoid/sim_freq`    | 50~200 Hz      | 调整仿真频率，数值越高实时性越好（需硬件支持，默认 100）        |
| `/humanoid/use_viewer`  | true/false     | 启用/禁用 Mujoco 可视化窗口（禁用可提升仿真性能）               |
| `/humanoid/init_wait_time` | 2.0~10.0 s   | 调整初始稳定时间，确保机器人启动后姿态平稳（默认 4.0）         |

## 常见问题解决

| 问题描述                                                     | 解决方案                                                     |
|--------------------------------------------------------------|--------------------------------------------------------------|
| 编译报错「The script 'humanoid_sim_node.py' doesn't exist」   | 修改 `setup.py`，确保 `scripts` 列表仅包含 `main.py`（参考项目结构中的配置） |
| 仿真启动报错「ValueError: The truth value of an array is ambiguous」 | 替换 `utils.py` 中的 `clip_value` 函数为 numpy 兼容版本（参考核心功能模块） |
| Mujoco 可视化窗口不显示                                       | 安装兼容版本可视化库：`pip3 install mujoco-viewer==0.1.7`     |
| ROS 话题无数据输出                                           | 检查 `ROS_MASTER_URI` 配置（默认 `http://localhost:11311`），重启 `roscore` 与仿真节点 |
| 机器人运动不稳定或跌倒                                       | 降低 CPG 频率（如 0.3 Hz）或振幅（如 0.3），增加初始稳定时间   |
=======
# 人形机器人 CPG 步态控制仿真系统
## 1. 项目概述
本项目基于 MuJoCo 物理仿真平台，实现了一套**基于 CPG（中枢模式发生器）的人形机器人步态生成与控制系统**。系统以仿生节律算法为核心，完成机器人双腿交替摆动的自然步态生成，并支持键盘实时控制、多模式步态切换、状态机管理与多级安全保护。

受限于简化机器人模型结构，项目重点验证 **CPG 步态生成算法有效性** 与**基础运动控制框架**，可为后续双足机器人平衡控制、重心规划、动态行走等高级算法提供可扩展的开发平台。

## 2. 功能
- **CPG 仿生节律步态**：模拟生物中枢神经模式，实现平滑、周期性的双腿交替运动
- **多状态控制逻辑**：STAND / WALK / STOP / EMERGENCY 状态管理
- **键盘实时控制**：W/S 行走启停，A/D 转向控制
- **多步态模式切换**：慢走 / 正常 / 快速 / 原地踏步
- **IMU 姿态传感器模拟**：获取机器人倾斜角与角速度
- **关节安全限位**：防止运动超限与姿态畸变
- **纯 Python 轻量化运行**：Windows 直接启动

## 3. 环境依赖
```bash
pip install mujoco numpy
- Python 3.8 ~ 3.11
- MuJoCo 2.3.0 及以上

## 4. 项目结构
├── models/
│   └── humanoid.xml       # MuJoCo 简化人形机器人模型
├── humanoid_stabilizer.py # 主控制程序（仿真入口）
├── cpg_oscillator.py      # CPG 中枢模式发生器
├── sensor_simulator.py    # IMU 姿态传感器模拟
├── keyboard_handler.py    # 键盘控制线程
├── utils.py               # 工具函数（数值裁剪、角度转换等）
└── README.md              # 项目说明文档
## 5. 核心模块说明
### 5.1 CPG 步态生成模块
- 采用正弦振荡器生成连续、周期性控制信号
- 输出信号直接驱动左右髋关节、膝关节交替运动
- 可在线调节频率、振幅，实现步态速度与步幅控制
### 5.2 运动控制模块
- 状态机控制机器人行为模式
- 行走状态：CPG 节律输出
- 停止/站立状态：关节归零，保持静止
- 紧急状态：控制量强制清零，安全停机
### 5.3 传感器模拟模块
- 模拟机器人躯干 IMU 传感器
- 输出滚动角(roll)、俯仰角(pitch)及其角速度
- 为后续平衡控制提供状态反馈
### 5.4 键盘控制模块
- 独立线程监听键盘输入
- 支持实时启停、转向、步态切换
- 操作简单、便于调试与演示

## 6. 使用方法
### 6.1 启动仿真
```bash
python humanoid_stabilizer.py

### 6.2 键盘控制说明
- `W`：进入行走模式，启动 CPG 步态
- `S`：停止步态，关节归零保持静止
- `A/D`：左右转向控制
- `1/2/3/4`：切换步态速度（慢走/正常/快速/踏步）
- `Q`：退出仿真

## 7. 改进方向
- 更换高精度双足机器人模型
- 加入 ZMP 稳定控制与倒立摆平衡算法
- 引入强化学习实现自适应步态优化
- 接入 ROS 实现多机通信与真实机器人部署
- 增加足底压力传感器、力控反馈
>>>>>>> upstream/main

## 参考资料
- [ROS 1 Noetic 官方文档](http://wiki.ros.org/noetic)
- [Mujoco 官方用户手册](https://mujoco.readthedocs.io/)
- [中枢模式发生器（CPG）步态控制综述](https://arxiv.org/abs/2003.02893)
- [ROS 包封装规范指南](http://wiki.ros.org/Packages)