ROS 2 PX4 深度强化学习无人机迷宫导航系统



\[项目简介]

本项目基于 ROS 2 (Humble) 与 PX4-Autopilot 仿真平台，使用 Stable Baselines3 框架训练无人机在 Gazebo 迷宫环境中自动寻路。

系统摒弃了传统的视觉处理，采用轻量化的 2D 激光雷达 (LaserScan) 数据作为输入，通过 PPO (Proximal Policy Optimization) 算法训练神经网络，实现从起点到设定终点的无碰撞自主飞行。针对虚拟机环境进行了专门的算力优化。



\[核心功能]



1\. ROS 2 环境封装

&nbsp;  通过 rclpy 构建自定义 Gym 环境，自动处理 ROS 节点通信。实现了无人机的自动解锁 (Arm)、模式切换 (Offboard) 以及起飞逻辑，无需人工干预仿真流程。



2\. 基于雷达的感知决策

&nbsp;  使用 180 点激光雷达数据作为观测空间 (Observation Space)，训练 MLP (多层感知机) 策略网络。相比图像输入，该方法对 CPU 算力要求极低，适合在虚拟机环境中快速训练。



3\. 安全与奖励机制

&nbsp;  防撞逻辑：当雷达检测到障碍物距离小于 0.3米 时给予高额惩罚并结束回合。

&nbsp;  引导奖励：根据无人机与目标点 (Target) 的欧氏距离实时计算奖励，引导无人机向终点靠近。

&nbsp;  高度锁定：通过 PID 逻辑强制锁定飞行高度为 2米，简化动作空间为二维平面的速度控制。



4\. 低配环境适配 (Headless Mode)

&nbsp;  支持无图形界面 (Headless) 运行 Gazebo，大幅降低显卡和 CPU 负载，解决虚拟机中 "Real Time Factor" 过低导致的仿真崩溃问题。



\[环境依赖]

运行本项目前，请确保已配置好 ROS 2 Humble、MAVROS 及 PX4-Autopilot 环境，并安装以下 Python 库：

pip3 install gymnasium stable-baselines3 shimmy numpy

pip3 install torch --index-url https://download.pytorch.org/whl/cpu



\[项目结构]

custom\_env.py : 自定义 Gym 环境核心代码。负责订阅 /scan (雷达) 和 /mavros/local\_position/pose (位置)，发布 /mavros/setpoint\_velocity/cmd\_vel\_unstamped (速度指令)，并处理服务调用。

train.py      : 训练脚本。配置 PPO 算法超参数，实例化环境并启动训练循环，支持模型自动保存。

1maze.world   : Gazebo 仿真使用的迷宫地图文件。



\[运行方式]

为了保证仿真同步性，请严格按照以下顺序在不同终端启动：



1\. 启动 Gazebo 仿真 (无头模式推荐)

&nbsp;  为了防止虚拟机卡顿导致飞控上锁，建议使用 Headless 模式：

&nbsp;  cd ~/桌面/prometheus\_px4

&nbsp;  HEADLESS=1 PX4\_SITL\_WORLD=/home/hex/桌面/maze/1maze.world make px4\_sitl gazebo



2\. 启动 MAVROS 通信桥梁

&nbsp;  等待 Gazebo 启动完毕后运行：

&nbsp;  ros2 launch mavros px4.launch fcu\_url:=udp://:14540@127.0.0.1:14580



3\. 启动 AI 训练

&nbsp;  确保终端位于代码所在目录 (如 .venv)：

&nbsp;  python3 train.py



\[注意事项]

\- 目标点设置：请根据 1maze.world 的实际出口位置，修改 custom\_env.py 中的 TARGET\_POS 坐标。

\- 仿真速率：如果遇到 "Time jump detected" 报错，可在启动 Gazebo 时添加 PX4\_SIM\_SPEED\_FACTOR=0.5 降低仿真速率。

\- 解释器选择：由于 ROS 2 库位于系统路径，运行 Python 脚本时请务必使用系统解释器 (/usr/bin/python3)，避免使用隔离的虚拟环境。

