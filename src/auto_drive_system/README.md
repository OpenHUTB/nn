
# CARLA 自动驾驶基础场景实践

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
│   └── data_collector.py            # 数据记录工具（TODO）
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
- CARLA 0.9.15
- Python 3.7

### 快速开始
```bash
# 静态障碍场景
python carla_da_static.py

# 动态障碍场景（基础版）
python carla_da_dynamic.py

# 动态障碍场景（多摄像头版）
python carla_da_dynamic_with_camera.py
```
