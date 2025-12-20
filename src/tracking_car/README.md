# CARLA 多目标跟踪与行为分析系统

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![CARLA Version](https://img.shields.io/badge/CARLA-0.9.14%2B-orange)](https://carla.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

一个基于 CARLA 仿真环境和 YOLOv8 的实时车辆多目标跟踪系统，支持 2D/3D 感知融合、车辆行为分析、多天气场景适配和数据记录功能。

## 🌟 核心功能

- 🚗 **实时目标检测与跟踪**：基于 YOLOv8 + SORT 算法实现车辆检测与多目标跟踪
- 🌤️ **多天气场景适配**：支持晴天、雨天、雾天、夜晚、多云、雪天等天气，并自动调整图像增强策略
- 📊 **车辆行为分析**：检测停车、超车、变道、刹车、危险接近等行为
- 📡 **多传感器融合**：支持 RGB 相机 + LiDAR 点云融合检测
- 📝 **数据记录**：自动记录跟踪结果、性能指标和配置参数
- 🎮 **3D 可视化**：实时显示 LiDAR 点云数据和跟踪结果
- ⚡ **高性能**：多线程架构，支持 GPU 加速和模型量化

## 📋 环境要求

### 基础环境
- Python 3.7
- CARLA 0.9.14 或更高版本
- CUDA 11.8+ (推荐，用于 GPU 加速)

### 硬件要求
- CPU：4核以上
- GPU：NVIDIA GPU (8GB 显存以上，推荐 RTX 3060+)
- 内存：16GB 以上

## 🛠️ 安装步骤

### 1. 安装 CARLA
参考 [CARLA 官方文档](https://carla.readthedocs.io/en/latest/start_quickstart/) 安装 CARLA 仿真环境：

```bash
# 方式1：使用 pip 安装
pip install carla

# 方式2：下载预编译版本
# https://github.com/carla-simulator/carla/releases
```

### 2. 安装依赖包
```bash
# 克隆仓库
git clone https://github.com/your-username/carla-object-tracking.git
cd carla-object-tracking

# 安装依赖
pip install -r requirements.txt
```

### 3. 依赖包列表
核心依赖包：
```txt
carla>=0.9.14
ultralytics>=8.0.0
torch>=2.0.0
opencv-python>=4.8.0
numpy>=1.24.0
open3d>=0.17.0
scipy>=1.10.0
scikit-learn>=1.2.0
numba>=0.58.0
loguru>=0.7.0
pyyaml>=6.0
psutil>=5.9.0
dataclasses>=0.6
```

## 🚀 快速开始

### 1. 启动 CARLA 服务器
```bash
# 进入 CARLA 安装目录
cd /path/to/carla/root

# 启动服务器
./CarlaUE4.sh -windowed -ResX=800 -ResY=600
```

### 2. 运行跟踪程序
```bash
# 基本运行
python carla_tracking.py

# 指定配置文件
python carla_tracking.py --config config.yaml

# 自定义参数
python carla_tracking.py --host localhost --port 2000 --conf-thres 0.5 --weather rain
```

### 3. 交互控制
| 按键 | 功能 |
|------|------|
| ESC  | 退出程序 |
| W/w  | 切换天气模式（晴天→雨天→雾天→夜晚→多云→雪天） |

## ⚙️ 配置说明

### 配置文件格式 (config.yaml)
```yaml
# 基础配置
host: "localhost"
port: 2000
num_npcs: 20

# 图像配置
img_width: 640
img_height: 480

# 检测配置
conf_thres: 0.5
iou_thres: 0.3
yolo_model: "yolov8n.pt"
yolo_imgsz_max: 320
yolo_iou: 0.45
yolo_quantize: false

# 跟踪配置
max_age: 5
min_hits: 3
kf_dt: 0.05
max_speed: 50.0

# 行为分析配置
stop_speed_thresh: 1.0
stop_frames_thresh: 5
overtake_speed_ratio: 1.5
overtake_dist_thresh: 50.0
lane_change_thresh: 0.5
brake_accel_thresh: 2.0
turn_angle_thresh: 15.0
danger_dist_thresh: 10.0
predict_frames: 10

# 可视化配置
window_width: 1280
window_height: 720
display_fps: 30
track_history_len: 20

# LiDAR 配置
use_lidar: true
lidar_channels: 32
lidar_range: 100.0
lidar_points_per_second: 500000
fuse_lidar_vision: true

# 数据记录配置
record_data: true
record_dir: "track_records"
record_format: "csv"
record_fps: 10
save_screenshots: false

# 3D 可视化配置
use_3d_visualization: true
pcd_view_size: 800
```

### 命令行参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| --config | 配置文件路径 | None |
| --host | CARLA 服务器地址 | localhost |
| --port | CARLA 服务器端口 | 2000 |
| --conf-thres | 检测置信度阈值 | 0.5 |
| --weather | 初始天气模式 | clear |

## 📁 输出文件结构

程序运行时会自动创建记录目录（默认：`track_records/`）：
```
track_records/
└── 20250101_120000/          # 时间戳目录
    ├── config.yaml           # 运行配置备份
    ├── performance.csv       # 性能指标记录
    ├── track_results.csv     # 跟踪结果记录
    └── screenshots/          # 截图目录（可选）
        └── screenshot_clear_000001.png
```

### 跟踪结果字段说明 (track_results.csv)
| 字段 | 说明 |
|------|------|
| timestamp | 时间戳 |
| frame_id | 帧ID |
| track_id | 跟踪ID |
| x1,y1,x2,y2 | 检测框坐标 |
| cls_id | 类别ID |
| cls_name | 类别名称 (Car/Bus/Truck/Unknown) |
| behavior | 行为标签 |
| speed | 估计速度 |
| confidence | 检测置信度 |

### 性能指标字段说明 (performance.csv)
| 字段 | 说明 |
|------|------|
| timestamp | 时间戳 |
| frame_id | 帧ID |
| fps | 帧率 |
| cpu_usage | CPU 使用率 (%) |
| memory_usage | 内存使用率 (%) |
| gpu_usage | GPU 使用率 (%) |
| detection_count | 检测目标数 |
| track_count | 跟踪目标数 |

## 🎨 可视化界面说明

### 主可视化窗口
- **顶部信息栏**：显示 FPS、天气、跟踪数量、行为统计、性能指标
- **检测框**：蓝色边框，显示类别、置信度、跟踪ID
- **行为标签**：红色背景显示 STOP/DANGER 等关键行为
- **轨迹线**：绿色线条显示车辆运动轨迹

### LiDAR 3D 窗口
- 实时显示点云数据，Z轴高度用颜色编码（红→蓝）
- 支持鼠标交互旋转/缩放视角

## 📚 核心算法说明

### 目标检测
- 使用 YOLOv8 作为基础检测器，支持 Car/Bus/Truck 三类车辆
- 针对不同天气自动调整图像增强策略（去雾、去雨、去雪、降噪）

### 多目标跟踪
- 基于 SORT 算法，使用卡尔曼滤波预测目标位置
- 匈牙利算法进行检测框匹配
- IOU 作为匹配代价

### 行为分析
- **停车**：速度低于阈值且持续多帧
- **超车**：相对自车速度比超过阈值
- **变道**：横向位移超过阈值
- **刹车**：加速度低于负阈值
- **危险接近**：距离自车过近

## 🔧 常见问题

### Q1: CARLA 连接失败
```
解决方法：
1. 确认 CARLA 服务器已启动
2. 检查端口是否正确（默认 2000）
3. 关闭防火墙或添加例外
```

### Q2: GPU 内存不足
```
解决方法：
1. 降低 yolo_imgsz_max 参数
2. 使用更小的模型（如 yolov8n.pt 而非 yolov8x.pt）
3. 启用 yolo_quantize: true 量化模型
```

### Q3: 帧率过低
```
解决方法：
1. 降低 display_fps 参数
2. 关闭 LiDAR (use_lidar: false)
3. 减少 NPC 数量 (num_npcs)
4. 降低图像分辨率 (img_width/img_height)
```

### Q4: 自车生成失败
```
解决方法：
1. 检查 CARLA 地图是否加载完成
2. 减少 NPC 数量避免碰撞
3. 程序会自动尝试偏移位置重试
```

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [CARLA Simulator](https://carla.org/) - 开源自动驾驶仿真平台
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - 目标检测模型
- [SORT Algorithm](https://github.com/abewley/sort) - 多目标跟踪算法