# 手势控制无人机项目

基于计算机视觉的手势识别无人机控制系统，支持本地仿真和 AirSim 真实模拟器，集成深度学习手势识别。

## 功能特性

- ✅ 实时手势识别（MediaPipe / OpenCV 双模式支持）
- ✅ 动态灵敏度调节（LOW/MEDIUM/HIGH 三档可调）
- ✅ 双手控制模式（左手方向 + 右手高度）
- ✅ 滑动手势控制（支持上下左右滑动）
- ✅ 本地 3D 仿真模式（OpenGL 渲染）
- ✅ AirSim 真实模拟器集成
- ✅ 传统机器学习手势分类（SVM、随机森林、MLP）
- ✅ **深度学习手势分类（CNN、Transformer、深度MLP）**
- ✅ **多模型对比训练**
- ✅ **实时可视化结果**
- ✅ 数据收集与训练

## 项目结构

```text
drone_hand_gesture/
├── main.py                    # 主程序（本地仿真模式）
├── main_airsim.py             # AirSim 真实模拟器版本
├── main_deep_learning.py      # 深度学习演示版本
├── airsim_controller.py       # AirSim 控制器
├── drone_controller.py        # 无人机控制器
├── simulation_3d.py           # 3D 仿真
├── physics_engine.py          # 物理仿真引擎
├── gesture_detector.py        # 基础手势检测器
├── gesture_detector_enhanced.py   # 增强手势检测器
├── gesture_classifier.py      # 传统ML手势分类器
├── deep_gesture_classifier.py    # 深度学习手势分类器
├── deep_gesture_detector.py      # 深度学习手势检测器
├── gesture_visualizer.py       # 手势可视化器
├── gesture_data_collector.py  # 手势图像数据收集
├── train_gesture_model.py     # 训练传统ML模型
├── train_deep_gesture.py      # 训练深度学习模型
└── requirements.txt           # 依赖列表
```

## 快速开始

### 方案 1：安装依赖

```bash
# 进入项目目录
cd src/drone_hand_gesture

# 安装依赖
pip install -r requirements.txt
```

### 方案 2：本地仿真模式

```bash
# 运行主程序
python main.py
```

### 方案 3：AirSim 真实模拟器模式

**前提条件**：
1. 安装 AirSim：`pip install airsim`
2. 运行 AirSim 模拟器（如 Blocks.exe）

```bash
# 运行 AirSim 版本
python main_airsim.py
```

### 方案 4：深度学习演示

```bash
# 训练深度学习模型
python train_deep_gesture.py --model_type cnn --epochs 100

# 运行深度学习演示
python main_deep_learning.py --model_path dataset/models/gesture_deep_cnn.pth --show_charts
```

**控制方式**：
- **手势控制**：张开手掌（起飞）、食指向上（上升）、握拳（降落）等
- **键盘控制**：空格（起飞/降落）、T（起飞）、L（降落）、H（悬停）、Q/ESC（退出）

## 深度学习功能


| 模型类型 | 说明 | 优势 |
|---------|------|------|
| CNN | 1D卷积神经网络 | 捕捉局部特征，计算效率高 |
| Transformer | 注意力机制模型 | 建模长距离依赖关系 |
| MLP | 深度多层感知器 | 简单高效，适合小规模数据 |

### 训练命令

```bash
# 训练单一深度学习模型
python train_deep_gesture.py --model_type cnn --epochs 100 --batch_size 32

# 训练所有模型进行对比
python train_deep_gesture.py --compare
```

### 可视化功能

- 📊 实时置信度折线图
- 📈 手势分类统计柱状图
- 🏷️ 手势信息面板
- 👆 手势图标显示
- 📐 3D关键点投影

## 手势命令映射

| 手势 | 命令 | 说明 |
|------|------|------|
| open_palm | 起飞 | 张开手掌 |
| closed_fist | 降落 | 握拳 |
| pointing_up | 上升 | 食指向上 |
| pointing_down | 下降 | 食指向下 |
| victory | 前进 | 胜利手势（V字） |
| thumb_up | 后退 | 大拇指向上 |
| thumb_down | 停止 | 大拇指向下 |
| ok_sign | 悬停 | OK手势 |
| rock | 左转 | 摇滚手势 |
| peace | 右转 | 和平手势 |
