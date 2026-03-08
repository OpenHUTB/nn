# 自动驾驶仿真学习环境（基于Carla与Python3.7，集成ROS）

本项目基于[MMAP-DRL-Nav](https://github.com/CV4RA/MMAP-DRL-Nav)的核心架构，实现了一套“多模态感知+深度强化学习（DRL）+ROS部署”的自动驾驶导航系统。项目分为**Win11环境下的机器学习核心模块**（模型训练、CARLA仿真交互）和**Ubuntu ROS环境下的节点封装模块**（实时感知决策），可完成从多传感器数据融合到车辆控制指令输出的全流程，适用于自动驾驶导航算法的学习、验证与二次开发。


## 项目核心目标
通过融合视觉图像、激光雷达（LiDAR）、IMU等多源异构传感器数据，利用注意力机制完成特征融合，结合深度强化学习（DQN）训练决策模型，最终通过ROS节点输出车辆线速度/角速度控制指令，在CARLA仿真环境中验证“避障、路径跟踪、目标导航”等核心导航能力。


## 环境准备

### 依赖库安装
```bash
# 安装Python 3.7（建议使用虚拟环境）
# 安装Carla 0.9.11客户端
pip install carla==0.9.11
# 安装辅助依赖库
pip install numpy opencv-python matplotlib vscode-debugpy
```
- `carla==0.9.11`：自动驾驶仿真平台核心库，提供车辆、环境与传感器模拟
- `python 3.7`：项目开发与运行的Python版本
- `numpy`：数值计算支持
- `opencv-python`：图像数据处理
- `matplotlib`：数据可视化工具
- `vscode-debugpy`：VSCode调试支持

### 开发环境配置
1. 下载并安装[Carla 0.9.11官方发行版](https://github.com/carla-simulator/carla/releases/tag/0.9.11)
2. 安装[VSCode](https://code.visualstudio.com/)并配置Python 3.7解释器
3. 推荐插件：Python、Pylance、Code Runner（提升开发效率）

# 自动驾驶车辆导航系统


## 核心模块

### 1. 感知模块
- `perception_module.py`：处理多传感器输入（IMU、相机图像、激光雷达）的神经网络模块，包括：
  - 用于视觉场景理解的语义分割网络
  - 基于1D卷积的激光雷达特征提取
  - 处理激光雷达数据的障碍物检测子网络
  - 输出场景信息、分割结果、里程计、障碍物和边界特征

- `ros/perception_module.py`：ROS优化的感知模块，具有：
  - 传感器数据扁平化与拼接处理
  - 基于线性层的特征融合
  - 使用`cv_bridge`实现ROS图像与张量的转换


### 2. 跨域注意力融合
- `attention_module.py`：实现`CrossDomainAttention`类用于多模态特征融合：
  - 将异质输入（4D/2D张量）调整为统一的3D格式`[batch, seq_len, features]`
  - 通过动态线性层对齐特征维度
  - 使用带残差连接和层归一化的多头自注意力块
  - 拼接并融合场景、分割、里程计、障碍物和边界数据的特征


### 3. 决策模块
- `decision_module.py`：核心决策网络，具有：
  - 生成动作分布（转向角、油门）的策略网络
  - 从融合特征估计状态价值的价值网络
  - 通过均值池化聚合时序特征

- `ros/decision_module.py`：兼容ROS的决策模块：
  - 用于生成控制指令的序列线性层
  - 将输出限制在安全范围内（线速度：0~2m/s，角速度：-1~1rad/s）
  - 返回字典形式的控制指令，便于转换为ROS消息


### 4. 自评估梯度模型
- `sagm.py`：实现`SelfAssessmentGradientModel`用于动作价值估计：
  - 处理拼接的状态-动作特征的全连接网络
  - 输出Q值以评估动作质量
  - 与actor-critic框架集成，用于强化学习更新


### 5. 系统集成
- `main.py`：定义`IntegratedSystem`类，整合所有模块：
  - 端到端流程：多模态输入→感知→注意力融合→决策→自评估
  - 支持MSE损失训练（策略回归+价值估计+Q值匹配）
  - 包含带设备管理（CPU/GPU）的训练/测试循环


## ROS集成
- `ros/CMakeLists.txt`：配置ROS功能包编译：
  - 指定Python 3.7环境及依赖（rospy、std_msgs、sensor_msgs、cv_bridge）
  - 将Python脚本安装到ROS环境
  - 设置Catkin的包含目录

- `ros/package.xml`：ROS功能包元数据，包含依赖和维护者信息

- `ros/perception_decision_node.py`：主ROS节点：
  - 订阅传感器话题（模拟或真实数据）
  - 按指定频率（默认10Hz）运行感知/决策流程
  - 发布控制指令到`/cmd_vel`话题

- `ros/ros_test_node.py`：模块验证测试节点：
  - 使用模拟传感器数据验证感知/决策工作流
  - 记录特征形状和控制指令日志


## 仿真与训练
- `carla_environment.py`：CARLA模拟器接口（兼容Gym）：
  - 管理模拟器连接、车辆生成和传感器设置
  - 配置交通管理器（兼容CARLA 0.9.11）以控制NPC行为
  - 提供观测空间（图像、激光雷达、IMU）和动作接口

- `dataloader.py`：数据加载工具：
  - `CarlaDataset`类用于处理模拟的CARLA数据
  - 生成图像、激光雷达、IMU和动作数据的批次

- `dqn_agent.py`：DQN强化学习实现：
  - 经验回放缓冲区
  - ε-贪婪动作选择
  - 带伽马折扣因子的目标网络更新

- `run_simulation.py`：CARLA中平滑 spectator 相机控制工具，使用线性插值实现


## 核心工作流程
1. **感知**：多传感器数据（图像/激光雷达/IMU）被处理为结构化特征
2. **融合**：跨域注意力将异质特征合并为统一表示
3. **决策**：策略网络从融合特征生成控制指令
4. **评估**：SAGM评估动作质量以指导学习
5. **ROS集成**：指令发布到车辆控制话题，实现实时执行

## 参考资料
- [Carla 0.9.11官方文档](https://carla.readthedocs.io/en/0.9.11/)
- [Python 3.7官方文档](https://docs.python.org/3.7/)
- [VSCode Python开发指南](https://code.visualstudio.com/docs/languages/python)
- [项目来源](https://github.com/CV4RA/MMAP-DRL-Nav)