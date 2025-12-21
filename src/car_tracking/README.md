 Carla 工具集
本目录包含两个独立的Carla脚本，可直接运行，互不依赖。


## 程序1：carla_2d_ground_truth.py
### 功能
将车辆3D包围盒投影为2D地面真值边界框并可视化，适用于目标检测算法验证与数据集采集。

### 环境要求
- Carla 版本：0.9.10~0.9.15
- Python 版本：3.7
- 依赖包：`opencv-python` `numpy` `carla`（版本需与Carla模拟器匹配）


### 操作说明
- 可视化窗口显示相机画面与车辆2D边界框
- 按 **q** 键退出程序

### 完整运行步骤
### 运行命令

```bash
# 第一步：启动对应版本的Carla模拟器
# 第二步：执行脚本
python car_2d_ground_truth.py
```


## 程序2：carla_viewer.py
### 功能
Carla 0.9.14低画质版专用脚本，实现车辆自动行驶+提前避让（车辆/障碍物/红绿灯），修复低配置设备进程卡顿问题，降低资源占用。

### 环境要求
- Carla 版本：仅限0.9.14低画质版
- Python 版本：3.7
- 依赖包：`opencv-python` `numpy` `carla==0.9.14`（可选`psutil`用于进程检测）

### 操作说明
- 可视化窗口显示摄像头画面，终端实时输出车速与避让提示
- 按 **q** 键退出程序

### 完整运行步骤
```bash
# 第一步：启动Carla 0.9.14低画质版模拟器
# 第二步：执行脚本
python main.py
```

## 程序3：car1.py
### 功能
基于Carla Actor ID实现轻量化车辆检测、跟踪与速度可视化，利用3D包围盒投影生成2D边界框，替代Deep SORT依赖，优化地图加载与仿真同步效率。

### 环境要求
- Carla 版本：0.9.10~0.9.15（兼容主流版本，避免非内置地图名）
- Python 版本：3.7
- 依赖包：`opencv-python` `numpy` `carla`（版本需与Carla模拟器匹配）

### 操作说明
- 可视化窗口显示相机画面、车辆2D边界框、Actor ID及实时速度
- 终端输出主车辆行驶速度
- 按 **q** 键退出程序
- 程序启动时自动清理现有NPC，生成50辆自动驾驶NPC模拟交通场景

### 完整运行步骤
```bash
# 第一步：启动对应版本的Carla模拟器（建议确认可用地图：如Town01/Town06）
# 第二步：执行脚本
python car1.py
```

### 操作说明
- 可视化窗口显示相机画面与车辆2D边界框
- 按 **q** 键退出程序

## 程序2：carla_low_quality_viewer.py
### 功能
Carla 0.9.14低画质版专用脚本，实现车辆自动行驶+提前避让（车辆/障碍物/红绿灯），修复低配置设备进程卡顿问题，降低资源占用。

### 环境要求
- Carla 版本：**仅限0.9.14低画质版**
- Python 版本：3.7
- 依赖包：`opencv-python` `numpy` `carla==0.9.14`（可选`psutil`用于进程检测）

### 运行命令
```bash
# 第一步：启动Carla 0.9.14低画质版模拟器
# 第二步：执行脚本
python main.py
```

### 操作说明
- 可视化窗口显示摄像头画面，终端实时输出车速与避让提示
- 按 **q** 键退出程序

# CARLA-DeepSORT-Vehicle-Tracking

## 项目介绍
基于CARLA仿真器和Deep SORT算法的2D车辆实时追踪程序，支持NPC生成、车辆追踪、关键信息显示，已修复文字反向、编码错误等问题。

## 环境依赖
- Python 3.7
- CARLA 0.9.10+
- 依赖库：`pip install numpy opencv-python scipy carla`

## 快速开始
1. 启动CARLA仿真器（建议低画质：`CarlaUE4.exe -quality-level=Low`）
2. 运行程序：`python main.py`
3. 操作：按`q`键退出

## 核心功能
- 自动生成自车和20辆NPC（自动驾驶）
- Deep SORT实时追踪车辆，显示追踪ID和类别
- 实时显示车速、追踪车辆数、地图名称
- 自动清理资源，避免残留进程

## 已修复问题
1. 图像文字/边界框反向
2. `mars-small128.pb`编码读取错误
3. Deep SORT API弃用警告
4. 外部依赖缺失问题
