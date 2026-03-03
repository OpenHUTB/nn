# 基于实时图像分类的无人机导航系统

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10-red.svg)](https://pytorch.org/)
[![AirSim](https://img.shields.io/badge/AirSim-1.8.1-brightgreen)](https://microsoft.github.io/AirSim/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本项目实现了一套基于实时图像分类的无人机自主导航系统。通过在 **AirSim 仿真环境** 中部署轻量化深度学习模型，无人机能够实时感知周围环境（检测人、车辆、树木、建筑物、道路、障碍物、降落区等七类目标），并基于视觉输入做出智能导航决策（避障、目标跟踪、自主降落等）。系统可在仿真中完成算法验证与参数调优，并可无缝迁移至实物无人机平台（如 F450 + Pixhawk + Jetson Nano）。

---

## 📦 特性

- **实时视频流处理**：支持 AirSim 仿真相机图像流，也可接入 USB 摄像头或视频文件进行测试。
- **深度学习目标检测**：基于轻量化 Faster R-CNN 模型（经通道剪枝与知识蒸馏），在普通 PC 上实现实时推理，mAP@0.5 达 0.82。
- **多类别环境感知**：可识别 7 类关键目标（人、车辆、树木、建筑物、道路、障碍物、降落区），支持自定义扩展。
- **智能导航决策**：采用改进的人工势场法，根据检测结果实时生成飞行控制指令（避障、跟踪、探索、降落）。
- **紧急状况处理**：模拟电池管理，低电量时自动返航或降落；通信中断时触发安全保护。
- **多传感器融合**（可选）：在实物平台中融合 IMU 与超声波数据，通过扩展卡尔曼滤波提升鲁棒性。
- **完整的 AirSim 仿真支持**：提供多场景（城市、森林、废墟、工业园区）与天气变化（雨、雪、雾、不同光照）的仿真环境，支持 PX4 软件在环（SITL）测试。
- **模块化代码结构**：各功能模块（数据采集、目标检测、导航决策、控制接口）解耦，易于调试与扩展。

---

## 🛠 硬件需求（实物平台可选）

- **无人机平台**：F450 机架，2212 920KV 电机，30A 电调，3S 锂电池
- **飞控系统**：Pixhawk 4（运行 PX4 固件）
- **边缘计算机**：NVIDIA Jetson Nano（4GB 版本，实物部署用）
- **视觉传感器**：Logitech C920 USB 摄像头 或 其他 USB 摄像头
- **辅助传感器**：超声波模块（如 HC-SR04），IMU（飞控内置）
- **通信模块**：数传电台，Wi-Fi 模块（用于地面站通信）

---

## 💻 软件依赖（仿真环境）

- **操作系统**：Windows 10 / 11 或 Ubuntu 20.04 LTS
- **仿真环境**：AirSim 1.8.1（Unreal Engine 4.27）
- **深度学习框架**：PyTorch 1.10，Torchvision 0.11
- **计算机视觉库**：OpenCV 4.5
- **其他**：NumPy，Matplotlib，tqdm，等

---

## 📥 安装步骤

### 1. 克隆本仓库

```bash
git clone https://github.com/yourname/uav_vision_navigation.git
cd uav_vision_navigation
```

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 3. 安装并配置 AirSim 仿真环境

- 下载并安装 **AirSim** 的 Windows 二进制包，或从源码编译（推荐使用预编译的 Blocks 环境快速开始）。  
  官方下载地址：[https://github.com/Microsoft/AirSim/releases](https://github.com/Microsoft/AirSim/releases)

- 解压后运行 `Blocks.exe` 或自定义的 Unreal 环境。

- 确保 AirSim 设置文件中启用了摄像头图像 API（默认已启用）。  
  可选：修改 `settings.json` 以配置天气、光照等。

### 4. 下载预训练模型

- 从 [Release 页面](https://github.com/yourname/uav_vision_navigation/releases) 下载 `best_model.pth`，放置于 `models/` 目录下。
- 若需在 CPU 上运行，脚本会自动处理；若使用 GPU，请确保已安装 CUDA 和 cuDNN。

---

## 🚀 使用方法

### 在 AirSim 仿真中运行

1. 启动 AirSim 环境（例如 `Blocks.exe`）。
2. 运行主程序（将自动连接 AirSim 并开始图像采集、检测与导航）：
   ```bash
   python scripts/main.py --sim True
   ```
   可选参数：
   - `--model_path`：指定模型路径
   - `--scene`：选择预设场景（如 `urban`, `forest`, `ruins`）
   - `--weather`：设置天气效果（`rain`, `fog`, `snow` 等）

3. 程序将实时显示检测结果与导航状态，并控制 AirSim 中的无人机执行相应动作。

### 使用本地视频文件或摄像头测试

```bash
python scripts/main.py --source 0          # 使用默认摄像头
python scripts/main.py --source video.mp4  # 使用视频文件
```

### 生成训练数据集

项目提供脚本从 AirSim 自动采集图像并生成标注（需配合 AirSim 的物体检测 API）：
```bash
python scripts/data_collection.py --save_dir ./datasets/airsim_data --num_samples 5000
```

### 模型训练

如需在自己的数据集上训练，请参考 `training/` 目录下的说明。

---

## 📁 项目结构

```
uav_vision_navigation/
├── models/                 # 预训练模型存放目录
├── scripts/                # Python 核心代码
│   ├── main.py             # 主程序入口
│   ├── detector.py         # 目标检测类（加载模型、推理）
│   ├── navigation.py       # 导航决策算法（人工势场法）
│   ├── airsim_client.py    # AirSim 客户端接口（图像获取与控制）
│   ├── data_collection.py  # 数据采集脚本
│   └── utils.py            # 工具函数（可视化、坐标转换等）
├── training/               # 模型训练代码与配置文件
├── config/                 # 配置文件（模型参数、导航参数等）
├── datasets/               # 数据集说明与预处理脚本
├── docs/                   # 文档与图片
├── requirements.txt        # Python 依赖列表
├── README.md               # 本文件
└── LICENSE                 # 许可证
```

---

## 📊 结果展示

- **目标检测性能**：在测试集上 mAP@0.5 达到 0.82，各类别精度与召回率详见 [docs/report.md](docs/report.md)。
- **导航测试**：在 AirSim 复杂场景下避障成功率 94%，目标跟踪平均误差 0.20 米。
- **实时性**：在配备 GTX 1060 的 PC 上，端到端延迟约 85ms（含图像采集、推理、决策），满足实时要求。

（更多检测效果图与仿真飞行视频请见 [docs/demo.md](docs/demo.md)）

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进项目。请确保代码风格符合 PEP8，并附上必要的测试。

---

## 📄 许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- 感谢老师的悉心指导。
- 本项目使用了以下公开数据集：VisDrone, UAVDT, Stanford Drone Dataset。
- 参考了 Faster R-CNN、YOLO 等经典目标检测算法。

---

**如有任何问题，请联系作者：** 周仁杰（2309040035@qq.com）