# 机械臂仿真控制项目
基于 MuJoCo 的六自由度机械臂运动控制与轨迹规划仿真平台

## 项目简介
本项目基于 MuJoCo 物理引擎，实现 6 自由度机械臂的运动控制、轨迹跟踪、工作空间分析及键盘交互控制。
所有代码轻量化、可直接运行。

模型文件：arm6dof_final.xml
控制末端：wrist3

## git push origin main1已实现功能
- 多关节同步 PID 位置控制
- 末端画圆、、8 字轨迹
- 6D 位姿控制（位置 + 姿态保持）
- 键盘控制机械臂
- 机械臂关节保护
 ## 环境依赖
``` bash
pip install mujoco numpy pynput glfw opencv-python