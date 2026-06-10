<p align="center">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/opencv-4.x-green" alt="OpenCV">
  <img src="https://img.shields.io/badge/license-MIT-brightgreen" alt="License">
</p>

<h1 align="center">🛸 手势控制无人机</h1>
<p align="center"><strong>基于计算机视觉的实时手势识别无人机控制系统</strong></p>
<p align="center">支持本地 3D 仿真 & AirSim 模拟器 · 集成深度学习 & 传统机器学习</p>

---

## 📋 目录

- [✨ 功能特性](#-功能特性)
- [📁 项目结构](#-项目结构)
- [🚀 快速开始](#-快速开始)
- [🎮 控制指南](#-控制指南)
- [🖐 手势映射](#-手势映射)
- [📊 飞行统计面板](#-飞行统计面板)
- [🧠 深度学习](#-深度学习)
- [🔧 环境兼容性](#-环境兼容性)
- [📦 依赖](#-依赖)

---

## ✨ 功能特性

### 🎯 核心控制
| 功能 | 说明 |
|------|------|
| 🖐 **实时手势识别** | MediaPipe / OpenCV 双模式，自动降级切换 |
| 🎚️ **动态灵敏度调节** | LOW / MEDIUM / HIGH 三档可调，界面实时显示 |
| 🤲 **双手控制模式** | 左手控制方向 + 右手控制高度 |
| 🔄 **滑动手势控制** | 支持上下左右滑动控制飞行 |
| 🔁 **摄像头镜像切换** | 一键翻转画面（`M` 键） |
| ✊ **握拳起飞检测** | 握拳松手自动起飞，智能降落检测 |

### 🧭 飞行辅助
| 功能 | 说明 |
|------|------|
| 🎬 **轨迹录制与回放** | 完整记录飞行路径，支持回放复盘（`R` / `P` 键） |
| 📍 **航点标记导航** | 一键标记航点，自动导航到目标位置（`N` 键） |
| 📊 **飞行统计面板** | 实时展示飞行时长、距离、速度、手势频率、电池消耗（`V` 键） |

### 🤖 AI 能力
| 功能 | 说明 |
|------|------|
| 🧬 **传统 ML 分类** | SVM / 随机森林 / MLP 三种模型可选 |
| 🧠 **深度学习分类** | CNN / Transformer / 深度 MLP，支持对比训练 |
| 📈 **实时可视化** | 置信度折线图、手势统计柱状图、3D 关键点投影 |

---

## 📁 项目结构

```
drone_hand_gesture/
│
├── 🎯 主程序
│   ├── main.py                      # 主程序（本地仿真模式）
│   ├── main_airsim.py               # AirSim 真实模拟器版本
│   └── main_deep_learning.py        # 深度学习演示版本
│
├── 🛸 控制与仿真
│   ├── drone_controller.py          # 无人机控制器
│   ├── airsim_controller.py         # AirSim 控制器
│   ├── simulation_3d.py             # 3D OpenGL 仿真渲染
│   └── physics_engine.py            # 物理仿真引擎
│
├── 🖐 手势识别
│   ├── gesture_detector.py          # 基础手势检测器
│   ├── gesture_detector_enhanced.py # 增强手势检测器（MediaPipe）
│   ├── gesture_detector_cv.py       # OpenCV 肤色检测器
│   ├── gesture_classifier.py        # 传统 ML 手势分类器
│   ├── deep_gesture_classifier.py   # 深度学习手势分类器
│   ├── deep_gesture_detector.py     # 深度学习手势检测器
│   └── gesture_visualizer.py        # 手势可视化器
│
├── 📊 数据与统计
│   ├── gesture_data_collector.py    # 手势图像数据收集
│   └── flight_statistics.py         # 飞行统计数据收集与展示
│
├── 🏋️ 模型训练
│   ├── train_gesture_model.py       # 训练传统 ML 模型
│   └── train_deep_gesture.py        # 训练深度学习模型
│
└── 📦 配置文件
    └── requirements.txt             # Python 依赖列表
```

---

## 🚀 快速开始

### 环境要求

- **Python**：3.8+
- **摄像头**：USB 摄像头或内置摄像头
- **GPU**（可选）：训练深度学习模型推荐使用 CUDA

### 1. 安装依赖

```bash
# 进入项目目录
cd src/drone_hand_gesture

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行本地仿真

```bash
python main.py
```

> 启动后将打开 3D 仿真窗口和摄像头画面，使用手势或键盘控制无人机。

### 3. 运行 AirSim 模拟器

**前提条件**：
1. 安装 AirSim：`pip install airsim`
2. 启动 AirSim 模拟器（如 Blocks.exe）

```bash
python main_airsim.py
```

### 4. 运行深度学习演示

```bash
# 训练深度学习模型
python train_deep_gesture.py --model_type cnn --epochs 100

# 运行演示
python main_deep_learning.py --model_path dataset/models/gesture_deep_cnn.pth --show_charts
```

---

## 🎮 控制指南

### ⌨️ 键盘快捷键

| 类别 | 按键 | 功能 |
|------|------|------|
| 🛫 飞行控制 | `空格` | 起飞 / 降落 |
| | `T` | 起飞 |
| | `L` | 降落 |
| | `H` | 悬停 |
| 🧭 方向移动 | `W` `S` | 前进 / 后退 |
| | `A` `D` | 左移 / 右移 |
| | `Q` `E` | 左转 / 右转 |
| | `↑` `↓` | 上升 / 下降 |
| 🎚️ 灵敏度 | `1` | 低灵敏度 |
| | `2` | 中灵敏度 |
| | `3` | 高灵敏度 |
| 🎬 录制回放 | `R` | 开始 / 停止录制飞行轨迹 |
| | `P` | 回放飞行轨迹 |
| 📍 航点 | `N` | 标记当前航点 |
| 🔄 显示 | `M` | 摄像头镜像切换 |
| | `V` | 飞行统计面板 开/关 |
| ❌ 退出 | `Q` / `ESC` | 退出程序 |

### 🖐 手势控制

| 手势 | 命令 | 示意图 |
|------|------|--------|
| 🖐 **张开手掌** | 起飞 | 五指张开 |
| ✊ **握拳** | 降落 | 握紧拳头 |
| ☝️ **食指向上** | 上升 | 食指朝上 |
| 👇 **食指向下** | 下降 | 食指朝下 |
| ✌️ **V 字手势** | 前进 | 胜利手势 |
| 👍 **大拇指向上** | 后退 | 拇指朝上 |
| 👎 **大拇指向下** | 停止 | 拇指朝下 |
| 👌 **OK 手势** | 悬停 | 拇指食指圈起 |
| 🤘 **摇滚手势** | 左转 | 食指小指伸出 |
| ✌️ **和平手势** | 右转 | 食指中指伸出 |

---

## 🖐 手势映射

> 技术层面对应关系

| 手势标识 | 对应命令 | 描述 |
|----------|----------|------|
| `open_palm` | 起飞 | 手掌完全张开 |
| `closed_fist` | 降落 | 手指完全握紧 |
| `pointing_up` | 上升 | 仅食指伸出朝上 |
| `pointing_down` | 下降 | 仅食指伸出朝下 |
| `victory` | 前进 | V 字形（食指+中指） |
| `thumb_up` | 后退 | 拇指朝上，其余握拳 |
| `thumb_down` | 停止 | 拇指朝下，其余握拳 |
| `ok_sign` | 悬停 | 拇指与食指成圈 |
| `rock` | 左转 | 摇滚手势（食指+小指） |
| `peace` | 右转 | 和平手势（食指+中指背面） |

---

## 📊 飞行统计面板

按 `V` 键打开实时飞行统计面板，显示以下数据：

| 统计类别 | 具体指标 |
|----------|----------|
| ⏱️ **时间** | 飞行时长、会话时长 |
| 📏 **距离** | 累计飞行距离、离起飞点最远距离 |
| ⚡ **速度** | 当前速度、最大速度、平均速度 |
| 🏔️ **高度** | 当前高度、最大高度 |
| 🔋 **电池** | 当前电量、最低电量、耗电率 |
| 🖐 **手势** | 各手势触发次数与频率排行 |
| 🎮 **命令** | 各命令执行次数与频率排行 |
| 🛫 **起降** | 起飞次数、降落次数 |

> 程序退出时自动打印完整的飞行统计报告到控制台。

---

## 🧠 深度学习

### 模型对比

| 模型 | 架构特点 | 适用场景 |
|------|----------|----------|
| **CNN** | 1D 卷积神经网络 | 捕捉局部特征，计算效率高，适合实时场景 |
| **Transformer** | 注意力机制模型 | 建模长距离依赖关系，准确率更高 |
| **深度 MLP** | 多层全连接网络 | 结构简单，训练快速，适合小规模数据 |

### 训练命令

```bash
# 训练单一模型
python train_deep_gesture.py --model_type cnn --epochs 100 --batch_size 32

# 对比训练所有模型
python train_deep_gesture.py --compare
```

### 可视化功能

- 📊 实时置信度折线图
- 📈 手势分类统计柱状图
- 🏷️ 手势信息面板
- 👆 手势图标叠加显示
- 📐 3D 手部关键点投影

---

## 🔧 环境兼容性

### Windows + Python 3.11 特别优化

本项目针对 **Windows + Python 3.11** 环境做了专项适配：

| 方案 | 检测器 | 说明 |
|------|--------|------|
| 🥇 优先方案 | **MediaPipe** | 完整手部 21 关键点检测，精度最高 |
| 🥈 降级方案 | **OpenCV 肤色检测** | 纯 OpenCV 实现（YCrCb 色彩空间），零额外依赖 |

- **自动切换**：MediaPipe 不可用时自动降级为 OpenCV 方案，无需手动配置
- **跨平台**：核心功能支持 Windows、macOS、Linux

---

## 📦 依赖

```text
opencv-python>=4.5.0
numpy>=1.21.0
mediapipe>=0.10.0          # 可选，自动降级
scikit-learn>=1.0.0        # 传统 ML 模型
torch>=1.10.0              # 深度学习模型
matplotlib>=3.5.0          # 可视化
airsim                     # AirSim 模式（可选）
PyOpenGL>=3.1.0            # 3D 仿真渲染
```

---

<p align="center">
  <sub>Made with ❤️ | 手势控制无人机项目</sub>
</p>
