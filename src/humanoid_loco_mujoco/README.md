# LocoMuJoCo的介绍
LocoMuJoCo 是一款专为全身控制而设计的模仿学习基准。
它具有多种环境，包括四足动物、类人生物和（肌肉）骨骼人体模型， 每个都提供了全面的数据集（每个人形机器人超过 22,000 个样本）。


它为研究人员提供了现成的仿生环境和数据，无需从零搭建物理仿真场景，可直接用于开发模仿学习算法。
支持多种生物形态的运动开发，包括四足动物（如狗）、两足动物（如人类）以及肌肉骨骼驱动的人体模型。
内置真实世界数据（如动作捕捉数据、地面实测数据），算法可直接基于这些数据学习 “如何像生物一样行走、奔跑”。

## 主要优势
- 支持 MuJoCo（单一环境）和 MJX（并行环境）
- 包括 12 个人形环境和 4 个四足环境，具有 4 个生物力学人体模型
- 干净的单文件 JAX 算法，用于快速基准测试（PPO、GAIL、AMP、DeepMimic）
- 将训练和环境组合到一个 JIT 编译函数中，实现闪电般的快速训练
- 🚀 超过 22,000 个动作捕捉数据集（AMASS、LAFAN1、原生 LocoMuJoCo）为每个人形机器人
- 重新定位 机器人到机器人重定向允许将任何现有数据集从一个机器人重新定位到另一个
- 机器人 强大的轨迹比较指标，包括动态时间扭曲和离散 Fréchet 距离，全部在 JAX
-  Gymnasium
- 接口中 内置域和地形随机化
- 模块化设计：定义、交换和重用组件，如观察类型、奖励函数、终端状态处理程序和域随机化文档

# 安装说明

##  安装

安装 MuJoCo 并设置环境变量。
```
cd loco-mujoco
pip install -e . 
```
##  依赖

- numpy<2.0
- scipy>=1.14.0
- mujoco==3.2.7
- mujoco-mjx==3.2.7
- jax (CPU version by default)
- flax
- gymnasium

##  数据安装说明

### 1.模型地址

https://github.com/robfiras/loco-mujoco/tree/master/loco_mujoco/models/unitree_h1

### 2.数据集百度网盘地址

链接: https://pan.baidu.com/s/1Viqlg9VYZBuKgj7TiVo-Tw 提取码: 68aa
下载好的数据集放在当前文件夹下

# 文件使用说明

## 文件结构
```
locomujoco
├── ros2                      #  ros2接口
│   ├── launch                #  launch文件# 使用mujoco进行的强化学习训练    
│   ├── mujoco_ros2_cpp       #  c++_ros2封装
│   |     ├── mujoco_ros2_sim.cpp  #  mujoco与ros2的通信
│   |     └─── robot_control.cpp   #  控制节点
|   └── mujoco_ros2_py        #  py_ros2封装
│        ├── mujoco_ros2_sim.py   #  mujoco与ros2的通信
│        └─── robot_control.py    #  控制节点
├──  main.py               # 使用locomujoco的简单示例
├──  mujoco.py             #使用mujoco的简单实例
├── robot_RL.py            # 强化学习算法

```

## 文件使用

### 1.运行locomujoco的简单示例

```
python main.py
```
### 2.运行mujoco简单示例

```
python mujoco.py
```
### 3.运行强化学习行走

```
python robot_RL.py
```

## MuJoCo ROS2 文件使用说明

### 1.创建功能包

```
cd ros2/src
ros2 pkg create mujoco_ros2 --build-type ament_python --dependencies rclpy my_interfaces
```

### 2.运行mujoco_ros2_sim节点

```
ros2 run mujoco_ros2 mujoco_ros2_sim
```

### 3.运行robot_control节点

```
ros2 run mujoco_ros2 robot_control
```

# 项目参考

- [MuJoCo](https://github.com/deepmind/mujoco) - 高性能物理引擎
- [ROS 2](https://github.com/ros2) - 机器人操作系统
- [LocoMuJoCo](https://loco-mujoco.readthedocs.io) - locomujoco官方参考文档