# 自动驾驶感知与控制系统研究 (基于 CARLA)

## 1. 项目简介
本项目旨在通过 Python 脚本与 CARLA 仿真环境进行深度交互，搭建一个基础的自动驾驶测试环境。项目将探索如何利用深度学习视觉算法和虚拟传感器数据，实现对仿真世界中动态物体的检测、环境感知以及基础的车辆控制。

## 2. 选题说明
* **参考开源项目:** [kamilkolo22/AutonomousVehicle](https://github.com/kamilkolo22/AutonomousVehicle)
* **重构思路:** 原项目部分模块对 Windows 系统兼容性较差，本项目提取其“视觉识别 + 传感器交互”的核心架构，在 Windows 环境下使用纯 Python 配合 PyTorch 进行完全重构，以确保跨平台的易用性和代码的可读性。

## 3. 开发运行环境
* **操作系统:** Windows 10/11
* **仿真平台:** HUTB CARLA_Mujoco_2.2.1
* **编程语言:** Python 3.8
* **核心框架:** PyTorch (支持 CUDA 加速), OpenCV
* **开发工具:** Visual Studio Code / Anaconda

## 4. 模块结构与入口
* 本模块的所有核心代码存放于 `src/carla_yolo_detection` 目录下。
* 模块的主程序入口为 `main.py`。

# [第2次提交] carla_yolo_detection: 实时感知与背景车流系统

## 1. 模块功能
本模块实现了自动驾驶视觉感知的基础闭环：
- **实时物体检测**: 集成 YOLOv5s 模型，实时识别 CARLA 环境中的车辆与行人。
- **背景交通流生成**: 利用 Traffic Manager 自动随机部署 30 辆背景车，模拟动态路况。
- **异步推理架构**: 优化了图像处理流程，通过回调截取最新帧，避免了深度学习推理导致的画面卡死，并支持实时 FPS 显示。
- **安全监听**: 挂载碰撞传感器 (Collision Sensor)，实时在终端发出碰撞预警。

## 2. 运行指南

### 步骤 1：启动 CARLA 模拟器
运行 `CarlaUE4.exe`，等待地图加载完毕。

### 步骤 2：配置 Python 环境
> **⚠️ 核心避坑**：本项目基于 HUTB CARLA_Mujoco_2.2.1，必须先手动安装模拟器自带的 `carla` 库，不能直接 pip install carla。

请在 Anaconda 环境（推荐 Python 3.8）中，**依次执行**以下命令：

1. **安装底层 CARLA API** (请将路径替换为你电脑上实际的 `.whl` 路径)：
   
   ```bash
   pip install D:\hutb\hutb_car_mujoco_2.2.1\PythonAPI\carla\dist\hutb-2.9.16-cp38-cp38-win_amd64.whl
   
   ```
   
2. 安装常规依赖库：
   ```pip install -r src/carla_yolo_detection/requirements.txt -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)```


3. **(可选) 开启 GPU 显卡加速**：
   如果你拥有 NVIDIA 显卡并希望获得 30+ 的流畅 FPS，请**务必额外执行**此命令覆盖安装 CUDA 版 Torch：
   
   ```bash
   pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
   

步骤 3：运行程序
请在项目根目录下执行核心脚本：
```python src/carla_yolo_detection/main.py```