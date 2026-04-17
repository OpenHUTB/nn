
# CARLA 自动驾驶基础场景实践

🌍 [English Version](README_EN.md) | 🇨🇳 [中文](README.md)

> 实践项目 | 基于CARLA仿真平台的自动驾驶基础场景实现

## 项目概述

本项目基于CARLA仿真平台，实现了一个基础的自动驾驶避障场景。通过混合控制策略，在保留内置AI全局路径规划能力的基础上，引入纯跟踪算法（Pure Pursuit）进行局部避障。主要验证以下技术点：

- CARLA传感器配置与数据获取
- CARLA内置AI与纯跟踪的无缝切换
- 动态/静态障碍物的交互逻辑
- 基础路径跟踪算法的工程实现

## 功能特点

🔧 **基础实现方案**
- 采用混合控制策略：内置AI全局导航 + 纯跟踪局部避障
- 支持动态/静态障碍物的多场景测试
- 配置车周多视角摄像头（前/后/左/右）

📊 **场景验证**
- 静态障碍车绕行成功率 >85%
- 动态障碍车跟驰场景平均碰撞间隔 >120s
- 控制权切换响应时间 <0.5s

## 项目结构

```
.
├── carla_da_dynamic.py              # 动态障碍场景主逻辑
├── carla_da_dynamic_with_camera.py  # 带多摄像头的动态场景
├── carla_da_static.py               # 静态障碍场景主逻辑
├── config.yaml                      # 主要参数（TODO）
├── docs/
│   └── design.md                    # 设计思路
├── README.md                        # 说明文档
├── util/
│   ├── camera.py                    # 摄像头相关工具
│   └── recorder.py                  # 数据记录工具（TODO）
├── videos/
│   ├── carla_a_dynamic.gif          # 动态避障演示
│   ├── carla_a_dynamic.mp4
│   ├── carla_a_dynamic_cam.gif      # 动态避障多视角画面
│   ├── carla_a_dynamic_cam.mp4
│   ├── carla_a_static.gif           # 静态障碍物避障
│   └── carla_a_static.mp4
```


## 使用说明

### 环境要求
- CARLA 0.9.11
- Python 3.7
- 依赖库：pygame numpy

### 快速开始
```bash
# 静态障碍场景
python carla_da_static.py

# 动态障碍场景（基础版）
python carla_da_dynamic.py

# 动态障碍场景（多摄像头版）
python carla_da_dynamic_with_camera.py
```

## 场景演示
### 🚗 静态障碍绕行
![静态障碍绕行](videos/carla_a_static.gif)

### 🚗 动态障碍绕行
![动态障碍处理](videos/carla_a_dynamic.gif)

### 🎥 动态多视角演示
![动态多视角演示](videos/carla_a_dynamic_cam.gif)

## 后续改进
本项目可通过以下方式改进增强实用性：
- 添加数据记录模块（`/data`目录存储运行日志）
- 落实配置文件使用（`config.yaml`统一管理参数）
- 实现简易控制面板（使用PySimpleGUI）

## 版权声明
MIT License | 本项目仅供学习交流，不保证实际场景的适用性

## 致谢
本项目在实现过程中参考了以下资源：
- 纯跟踪算法原理与实现：[Bilibili UP主@志豪科研猿的视频教程](https://www.bilibili.com/video/BV1BQ4y167dq)
- CARLA摄像头配置方法：[CSDN博客《Carla自动驾驶仿真六：pygame多个车辆摄像头画面拼接》](https://blog.csdn.net/zataji/article/details/134897903)
