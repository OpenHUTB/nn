# MuJoCo ROS 2 Demo

这是一个基于MuJoCo物理引擎和ROS2的机器人仿真演示项目。该项目实现了一个人形机器人模型的物理仿真，并通过ROS2节点发布和订阅关节角度数据。

## 演示视频

https://private-user-images.githubusercontent.com/221759988/504662321-9983c539-6cb3-480d-b85e-83649e2c78c8.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjEyMTM2ODksIm5iZiI6MTc2MTIxMzM4OSwicGF0aCI6Ii8yMjE3NTk5ODgvNTA0NjYyMzIxLTk5ODNjNTM5LTZjYjMtNDgwZC1iODVlLTgzNjQ5ZTJjNzhjOC5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUxMDIzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MTAyM1QwOTU2MjlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1hOTQxODNlZjVhYWExZDZlNTBkZTg0YmI1N2E2OWM0NzdiMWJhOGUwNTgzMDBkNTA1ODUzOGIwOTVkMjRmZjFkJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.U2xyKRU3qZ4D-nL3fc38U9AjBZqKHnsLkVNASP4KfLY

## 项目概述

本项目包含以下主要组件：
- 使用MuJoCo物理引擎加载和仿真人形机器人模型
- Publisher节点定期发布机器人关节角度数据
- Subscriber节点接收并处理关节角度数据
- 完整的ROS2 launch文件用于启动整个系统

## 系统要求

- Ubuntu 22.04 LTS
- ROS 2 Humble
- Python 3.10
- Conda 环境管理器
- MuJoCo 物理引擎

## 安装步骤

### 1. 安装 ROS 2 Humble

按照官方指南安装 ROS 2 Humble：
https://docs.ros.org/en/humble/Installation.html

### 2. 创建 Conda 环境

```bash
# 创建并激活 conda 环境
conda create -n mjoco_py310 python=3.10
conda activate mjoco_py310

# 安装 mujoco、numpy等缺少的依赖包
pip install mujoco
```
## 编写文件

### 1.创建src目录
```bash
mkdir -p mujoco_ros_demo/src
cd mujoco_ros_demo/src
```

### 2.使用apache协议创建功能包
```bash
ros2 pkg create --build-type ament_python mujoco_ros_demo --dependencies std_msgs 
```

### 3.创建launch、config目录以及两个话题订阅发布文件，删除非必须文件（test/）
```bash
mkdir -p launch config
rm -rf test
```

### 4.编写文件，配置setup.py(注意版本的兼容问题)
```bash
mujoco_ros_demo/
├── config/
│   ├── assets/           # 3D模型文件(STL格式)
│   └── humanoid.xml      # 机器人模型配置文件
├── launch/
│   └── main.launch.py    # ROS2启动文件
├── mujoco_ros_demo/
│   ├── __init__.py
│   ├── mujoco_publisher.py   # 发布关节角度数据的节点
│   └── data_subscriber.py    # 订阅并处理数据的节点
├── package.xml           # ROS2包配置文件
├── setup.py             # Python包安装配置
└── README.md
```

### 5.模型文件说明
为保持代码仓库的简洁性并满足存储限制，本项目仅包含运行程序所必需的核心模型文件(共9个文件)。完整的人体模型文件集包含40多个3D部件文件，主要用于手部精细建模。

完整模型文件集（包含手部精细模型等）可通过以下链接下载：
[完整模型文件集网盘链接](https://pan.baidu.com/s/1SN5SWpyfKR7KYDbE8lzvBw?pwd=9e9u)

下载完整文件集后，将其解压到 `config/assets/` 目录中，可获得更完整的人体模型可视化效果。


### 6.节点说明

**MujocoPublisher** (mujoco_publisher.py)
功能: 加载MuJoCo人形机器人模型并发布关节角度数据
订阅主题: 无
发布主题: /joint_angles(std_msgs/Float64MultiArray)
参数: model_path: 机器人模型文件路径

**DataSubscriber** (data_subscriber.py)
功能: 订阅关节角度数据并计算平均角度值
订阅主题: /joint_angles(std_msgs/Float64MultiArray)
发布主题: 无
输出: 在终端打印接收到的关节角度和平均角度值


## 运行文件
```bash
colcon build
source install/setup.bash
ros2 launch mujoco_ros_demo main.launch.py
```

## 运行结果

### **1.使用ros2 node命令查看详细信息**
https://private-user-images.githubusercontent.com/221759988/504662359-96d717f3-816d-45ec-937d-a5afacb8ad6d.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjEyMTM2ODksIm5iZiI6MTc2MTIxMzM4OSwicGF0aCI6Ii8yMjE3NTk5ODgvNTA0NjYyMzU5LTk2ZDcxN2YzLTgxNmQtNDVlYy05MzdkLWE1YWZhY2I4YWQ2ZC5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUxMDIzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MTAyM1QwOTU2MjlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1jZTcxMzhhOTMyNzljMTBiODAxM2M0NzA4NGI3ODk2OTQ1MTExMWUxMzE5YjQ2ZmExZjM0OTMzYjA0MDFjZmY1JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.GP_uheocaR_SA7KHGhIlwWpH4BhHK6Atcmx2zJ5irgM

### **2.使用rqt可视化工具**
![rqt可视化工具截图](https://private-user-images.githubusercontent.com/221759988/504847607-c554883e-9f47-4760-84c0-d920b9080afb.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjEyMzg1MDgsIm5iZiI6MTc2MTIzODIwOCwicGF0aCI6Ii8yMjE3NTk5ODgvNTA0ODQ3NjA3LWM1NTQ4ODNlLTlmNDctNDc2MC04NGMwLWQ5MjBiOTA4MGFmYi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUxMDIzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MTAyM1QxNjUwMDhaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT02NGQ5ZjdiMjg0MjQ2ZTYxZTg1NTc1YTlkZGFjZGNiZmIzOWY5NGI5ZDMxNjBhZWE4MjM1YmViZWIxNWNkZjZkJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.kJVbj-1bJWBL2AJChdFY8trfH9TrITbZnN_CdCDjGO8)



## 许可证
本项目基于Apache-2.0许可证发布。

## 贡献参考
- [MuJoCo](https://github.com/deepmind/mujoco) - 高性能物理引擎
- [ROS 2](https://github.com/ros2) - 机器人操作系统
- [User-in-the-Box](https://github.com/User-in-the-Box/user-in-the-box) - 项目参考