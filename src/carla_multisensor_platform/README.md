# 🏎️ CARLA 多传感器仿真平台

基于 CARLA 模拟器的自动驾驶仿真平台，支持多传感器数据采集、实时可视化、车道检测和数据集记录。

## 📌 目录

- [功能特点](#功能特点)
- [项目结构](#项目结构)
- [运行环境](#运行环境)
- [安装步骤](#安装步骤)
- [使用方法](#使用方法)
- [传感器配置](#传感器配置)
- [数据记录](#数据记录)
- [键盘控制](#键盘控制)
- [常见问题](#常见问题)
- [许可证](#许可证)

## ✨ 功能特点

- **多传感器支持**：RGB相机、深度相机、激光雷达、语义分割、实例分割
- **实时车道检测**：基于 YOLOPv2 的车道和可行驶区域检测（GPU加速）
- **数据记录系统**：同步记录图像、控制信号和车辆状态，可配置采样率（5-10 Hz）
- **鹰眼地图**：仿真环境的鸟瞰视图可视化
- **交通模拟**：自动生成车辆和行人
- **天气控制**：可配置天气条件
- **键盘控制**：支持手动控制车辆
- **同步模式**：确定性仿真，保证数据采集一致性

## 📁 项目结构

├── main.py # 主程序入口（已重命名）
├── requirements.txt # Python 依赖列表
├── Sensors/ # 传感器模块
│ ├── SensorManager.py # 传感器初始化与管理
│ ├── SensorHandler.py # 传感器数据回调处理
│ ├── RGBcamera/ # RGB相机相关
│ │ ├── YOLOPv2Detecor.py # YOLOPv2 车道检测
│ │ └── CarLaneDetector.py # 车辆和车道检测
│ └── Lidar/ # 激光雷达
│ └── lidar.py
├── utils/ # 工具模块
│ ├── environment.py # CARLA 环境配置
│ ├── DisplayManager.py # Pygame 显示管理
│ ├── DataRecorder.py # 数据集记录系统
│ ├── EgoVehicleController.py # 车辆控制
│ ├── eagle_eye_map.py # 鸟瞰地图
│ └── weather.py # 天气控制
├── dataset/ # 录制的驾驶数据
└── docs/ # 文档```


## 💻 运行环境

| 组件 | 版本要求 |
|------|----------|
| 操作系统 | Windows 10/11 或 Ubuntu 20.04+ |
| CARLA 模拟器 | 0.9.15 |
| Python | 3.8+ |
| GPU（推荐） | 支持 CUDA 的 NVIDIA 显卡 |

### Python 依赖

```bash
numpy==1.21.6
opencv-python==4.5.5.64
pygame==2.1.2
transforms3d==0.4.1
colorama==0.4.6
carla==0.9.15
torch>=1.7.0      # YOLOPv2 需要
torchvision>=0.8.0

##🚀 安装步骤
1. 安装 CARLA 模拟器
从 CARLA 官方 GitHub 下载 CARLA 0.9.15

解压到任意目录（例如 D:\CARLA）

注意：不要解压到包含中文或空格的路径

2. 安装 Python 依赖
bash
pip install -r requirements.txt
3. 下载 YOLOPv2 预训练模型（可选，用于车道检测）
下载 yolopv2.pt 模型文件

放置在 Sensors/RGBcamera/model/pretrained/ 目录下

如果不需要车道检测，可以在配置中禁用相关传感器

4. 配置 CARLA 路径（重要）
打开 main.py，找到被注释的 CARLA 路径配置，取消注释并修改为你的实际路径：

python
carla_egg = glob.glob('D:/CARLA/WindowsNoEditor/PythonAPI/carla/dist/carla-*.egg')
if carla_egg:
    sys.path.append(carla_egg[0])

##🎮 使用方法
启动仿真
第1步：启动 CARLA 服务器

bash
# Windows
双击 CarlaUE4.exe

# Linux
./CarlaUE4.sh
看到城市景观窗口后，保持运行。

第2步：运行仿真程序

打开新的命令行窗口：

bash
python main.py
一键启动（可选）
创建 main.bat（Windows）：

batch
@echo off
echo 正在启动 CARLA 仿真...
python main.py
pause

##📷 传感器配置
在 main.py 中修改 sensors_dict：

python
sensors_dict = {
    'RGBCamera': [[0, 0, 2.4], [0, 1], True],   # [位置, 网格位置, 是否启用]
    'DepthCamera': [[0, 0, 2.4], [0, 0], False],
    'LiDAR': [[0, 0, 2.4], [1, 0], False],
    # ...
}
传感器	描述	默认状态
RGBCamera	前置 RGB 相机	启用
DepthCamera	深度相机	禁用
LiDAR	三维激光雷达	禁用
SemanticSegmentationCamera	语义分割	禁用

## 💾 数据记录
系统自动记录以下数据：

图像：RGB 相机帧（400×224 JPG）

控制信号：转向、油门、刹车

车辆状态：速度、位置、旋转角

时间戳：帧 ID 和时间信息

输出格式
text
dataset/
└── session_YYYYMMDD_HHMMSS/
    ├── images/
    │   └── frame_XXXXXX.jpg
    ├── metadata/
    │   └── frame_XXXXXX.json
    └── session_summary.json


⌨️ 键盘控制
按键	        功能
R	        开始/停止数据记录
S	        显示录制状态
ESC    	退出仿真
W / ↑	加速
S / ↓  	刹车
A / ←	左转
D / →	右转
空格	        手刹
❓ 常见问题
1. 运行报错 ModuleNotFoundError: No module named 'carla'
原因：CARLA Python API 未正确安装

解决：
bash
# 方法一：使用 CARLA 自带的 egg 文件
cd D:\CARLA\WindowsNoEditor\PythonAPI\carla\dist
easy_install carla-0.9.15-py3.8-win-amd64.egg

# 方法二：安装 pip 包
pip install carla
2. 报错 timeout waiting for simulator
原因：CARLA 服务器未启动或端口不对

解决：
确保已双击 CarlaUE4.exe

检查代码中的端口：client = carla.Client('localhost', 2000)

3. 车道检测不工作
原因：缺少 YOLOPv2 预训练模型

解决：

下载 yolopv2.pt 放入指定目录

或在 sensors_dict 中禁用车道检测相机

4. Git 推送失败（网络问题）
bash
# 取消代理
git config --global --unset http.proxy
git config --global --unset https.proxy
📝 许可证
本项目仅供研究和教育用途。

## 🎬 运行效果

### 终端运行日志
以下为 `main.py` 成功运行时的终端输出，证明程序已正确连接 CARLA 服务器并初始化所有模块：

CARLA 0.9.15 connected at 127.0.0.1:2000.
INFO:  Found the required file in cache!  Carla/Maps/Nav/Town10HD_Opt.bin
INFO:  Found the required file in cache!  Carla/Maps/TM/Town10HD.bin
Initializing sensor: RGBCamera

AUTONOMOUS DRIVING DATA RECORDING SYSTEM
============================================================
Controls:
  R - Toggle data recording ON/OFF
  S - Show recording status
  ESC - Exit simulation

Data being recorded:
  - RGB camera images (400x224)
  - Control signals (steer, throttle, brake)
  - Vehicle speed and transform
  - Timestamps and frame IDs
  - Sampling rate: 10.0 Hz
  - Output directory: dataset
============================================================

## 🔧 修复记录

本项目基于原 `Carla-Project` 进行了以下修复和优化：

| 问题 | 修复方案 |
|------|----------|
| `EgoVehicleController` 缺少 `controller` 属性 | 在 `__init__` 中初始化 `self.controller = None`，并在 `setup_ego_vehicle` 中正确赋值 |
| YOLOPv2 模型缺失 | 添加 `yolopv2.pt` 预训练模型文件 |
| 入口文件名不符合规范 | 将 `Main.py` 重命名为 `main.py` |

## ⚠️ 硬件要求

- **最低要求**：独立显卡（6GB+ 显存）
- **集成显卡**：只能运行无渲染模式（`-RenderOffScreen`），无法显示 3D 画面

## 📝 已知问题

- 集成显卡环境下，CARLA 3D 城市窗口无法正常显示
- 建议在独立显卡环境下运行以获得完整可视化体验

## 🚀 一键启动

### Windows
双击 `main.bat` 即可启动

### Linux/Mac
```bash
chmod +x main.sh
./main.sh

🤝 贡献
欢迎提交 Issue 和 Pull Request。