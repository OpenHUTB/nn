# V2X 路侧智能感知系统

## 项目概述

基于 CARLA 0.9.16 仿真平台和 YOLOv8n 预训练目标检测模型，实现路侧单元（RSU）的智能感知功能。系统模拟真实 V2X（Vehicle-to-Everything）场景中路侧摄像头对交通参与者的实时检测、统计与预警信息推送。

**核心思路**：将轻量级目标检测模型部署在路侧边缘计算设备上，实时感知道路交通状态（车辆、行人），并通过 V2X 通信向附近车辆推送预警信息（行人提醒、拥堵警告、天气提示），提升交通安全。

## 核心功能

### 1. 路侧摄像头部署
- 在道路旁高处（8米）架设 RGB 摄像头，模拟真实交通监控视角
- 支持可配置的摄像头位置、俯仰角度和视场角

### 2. YOLOv8n 实时目标检测
- 使用 COCO 预训练模型，**无需训练**直接推理
- 支持检测：车辆（car/truck/bus/motorcycle）、行人（person）、交通标志等 80 类目标
- 实时绘制检测框、类别标签和置信度分数

### 3. V2X 信息面板
- 路侧单元（RSU）信息面板实时展示
- **检测统计**：车辆数量、行人数量、总目标数
- **V2X 预警广播**：行人预警、拥堵提醒、天气驾驶建议
- 仿真时间戳和帧率监控，模拟真实监控画面样式

### 4. 多天气场景切换
- 支持 7 种天气：晴天、多云、小雨、暴风雨、大雾、黄昏、夜晚
- 按 W 键实时切换，可观察不同天气条件对检测效果的影响

### 5. 动态交通流
- 自动生成 NPC 车辆（40辆）和行人（20人）
- NPC 自动驾驶/行走，构成真实动态交通场景

## 环境要求

### 软件
- Python 3.8+
- CARLA 0.9.16（需先启动 CarlaUE4.exe）

### 依赖安装

```bash
pip install -r requirements.txt
```

或手动安装：

```bash
pip install carla opencv-python numpy ultralytics
```

## 运行方法

### 1. 启动 CARLA 服务器

```bash
CarlaUE4.exe
```

### 2. 运行感知系统

```bash
cd src/edge_intelligence_V2X
python main.py
```

### 按键操作

| 按键 | 功能 |
|------|------|
| **W** | 切换天气场景（晴→云→雨→暴风→雾→黄昏→夜） |
| **S** | 保存当前画面截图到 `results/` 目录 |
| **Q / ESC** | 退出系统 |

## 项目结构

| 文件 | 说明 |
|------|------|
| `main.py` | 系统主程序入口 |
| `requirements.txt` | Python 依赖列表 |
| `README.md` | 项目说明文档 |
| `results/` | 截图保存目录（运行后自动生成） |
| `src/` | 原始参考代码 |

## 可调参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CAMERA_HEIGHT_M` | 8.0 | 摄像头架设高度（米） |
| `CAMERA_FOV` | 100 | 摄像头视场角（度） |
| `CAMERA_PITCH` | -25.0 | 摄像头俯仰角（负值向下） |
| `CAMERA_SPAWN_INDEX` | 0 | 参考路点索引（切换部署位置） |
| `NPC_VEHICLE_COUNT` | 40 | NPC 车辆数量 |
| `NPC_WALKER_COUNT` | 20 | NPC 行人数量 |
| `YOLO_CONFIDENCE` | 0.35 | 检测置信度阈值 |

## 技术参考

- [CARLA 仿真平台文档](https://carla.readthedocs.io/)
- [YOLOv8 目标检测 (Ultralytics)](https://docs.ultralytics.com/)
- [V2X 车路协同技术](https://en.wikipedia.org/wiki/Vehicle-to-everything)
- [COCO 数据集类别](https://docs.ultralytics.com/datasets/detect/coco/)
