# YOLOv8 图像目标检测系统

本项目基于 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) 构建，提供一个功能丰富、模块化的目标检测工具，支持多种检测模式和扩展功能。

***

## ✨ 功能特性

### 核心检测功能

- 🖼️ **静态图像检测** - 单张图像目标检测
- 📹 **实时摄像头检测** - 实时视频流目标检测
- 📦 **批量图像检测** - 批量处理图像文件夹
- 🎬 **视频文件检测** - 处理视频文件

### 扩展功能

- ⚖️ **多模型对比测试** - 同时加载多个模型进行性能对比
- 📊 **统计分析模块** - 检测结果统计分析与报告生成
- 📝 **命令行参数支持** - 支持脚本化运行
- 💾 **结果保存功能** - 支持多种格式结果保存
- 📋 **日志系统** - 完善的日志记录功能

### 增强功能

- 🚀 **自动设备选择** - 自动检测GPU/CPU
- 🔄 **目标跟踪** - 支持目标ID跟踪
- 📐 **距离估算** - 基于检测框的距离估算
- ⚠️ **危险等级评估** - 根据距离判定危险等级

***

## 📦 安装指南

### 环境要求

- Python 3.8 或更高版本
- pip 包管理器
- （可选）摄像头设备（用于实时检测）

### 安装步骤

#### 1. 进入项目目录

```bash
cd src/image_object_detection
```

#### 2. 安装依赖库

```bash
pip install -r requirements.txt
```

#### 3. 验证安装

```bash
python test_detection.py
```

***

## 📁 项目目录结构

```
image_object_detection/
│
├── main.py                     # 程序入口文件
├── requirements.txt            # Python 依赖列表
├── config.py                   # 全局配置管理（增强版）
├── config.yaml                 # 配置文件（YAML格式）
├── detection_engine.py         # YOLO 模型封装核心（增强版）
├── ui_handler.py               # 用户交互逻辑（优化版）
├── camera_detector.py          # 实时摄像头检测器
├── batch_detector.py           # 批量图像检测器（增强版）
├── model_manager.py            # 模型热切换管理器
├── video_detector.py           # 视频文件检测器
├── model_comparison.py         # 多模型对比测试模块
├── stats_analyzer.py           # 统计分析模块
├── result_saver.py             # 结果保存管理器
├── logger.py                   # 日志系统
│
├── test_detection.py           # 静态图像检测测试脚本
├── test_camera.py              # 摄像头检测测试脚本
├── test_model_comparison.py    # 多模型对比测试脚本
├── test_cli_args.py            # 命令行参数测试脚本
├── test_config.py              # 配置文件测试脚本
├── test_engine.py              # 检测引擎测试脚本
├── test_logger.py              # 日志系统测试脚本
├── test_result_saver.py        # 结果保存测试脚本
├── create_test_data.py         # 测试数据集生成脚本
│
├── data/                       # 测试数据目录
│   ├── input/                  # 输入图像目录
│   ├── output/                 # 输出结果目录
│   ├── test_images/            # 测试图像集
│   │   ├── test_single_person.jpg
│   │   ├── test_multiple_objects.jpg
│   │   ├── test_empty.jpg
│   │   ├── test_crowd.jpg
│   │   └── test_vehicles.jpg
│   └── README.md               # 数据集说明
│
└── README.md                   # 本说明文件
```

***

## 🚀 快速开始

### 方式一：交互式菜单运行（推荐）

```bash
python main.py
```

交互式菜单包含：

```
📋 主菜单
─────────────────────────────────────
  1. 📷 单图像检测
  2. 📹 摄像头实时检测
  3. 📁 批量图像检测
  4. 🎬 视频文件检测
  5. ⚖️ 多模型对比
  6. ⚙️ 设置
  7. 📊 统计分析
  8. ℹ️ 系统信息
  0. ❌ 退出
─────────────────────────────────────
```

### 方式二：命令行直接运行

```bash
# 单图像检测
python main.py --image ./data/test.jpg

# 摄像头检测
python main.py --camera

# 批量检测
python main.py --batch ./images --output ./results

# 多模型对比
python main.py --compare --models yolov8n.pt yolov8s.pt
```

***

## 📖 使用教程

### 1. 静态图像检测

选择菜单 `1`，可以：

- 使用默认测试图像
- 输入自定义图像路径
- 浏览目录选择图像

### 2. 实时摄像头检测

选择菜单 `2`：

- 打开摄像头窗口
- 按 `q` 键退出
- 实时显示检测框、距离和危险等级

### 3. 批量图像检测

选择菜单 `3`：

- 输入图像目录路径
- 自动检测所有支持格式的图像
- 结果保存到输出目录

### 4. 视频文件检测

选择菜单 `4`：

- 输入视频文件路径
- 可选保存输出视频

### 5. 多模型对比测试

选择菜单 `5`：

- 选择要对比的模型组合
- 自动运行对比测试
- 生成对比报告

### 6. 设置菜单

选择菜单 `6`：

- 切换检测模型
- 设置置信度阈值
- 设置默认路径
- 查看当前配置

***

## ⚙️ 命令行参数

| 参数          | 类型    | 说明      | 默认值        |
| ----------- | ----- | ------- | ---------- |
| `--image`   | str   | 单图像检测   | -          |
| `--camera`  | flag  | 摄像头检测   | -          |
| `--batch`   | str   | 批量检测目录  | -          |
| `--video`   | str   | 视频文件检测  | -          |
| `--compare` | flag  | 多模型对比模式 | -          |
| `--model`   | str   | 指定模型    | yolov8n.pt |
| `--conf`    | float | 置信度阈值   | 0.25       |
| `--models`  | list  | 多模型列表   | -          |
| `--output`  | str   | 输出路径    | -          |
| `--stats`   | flag  | 生成统计报告  | -          |
| `--quiet`   | flag  | 静默模式    | -          |
| `--debug`   | flag  | 调试模式    | -          |
| `--help`    | flag  | 显示帮助信息  | -          |

### 命令行使用示例

```bash
# 检测单张图像
python main.py --image ./data/test.jpg --model yolov8s.pt --conf 0.5

# 批量检测并生成统计
python main.py --batch ./images --output ./results --stats

# 多模型对比测试
python main.py --compare --models yolov8n.pt yolov8s.pt yolov8m.pt

# 摄像头检测，指定设备
python main.py --camera --cam-index 0
```

***

## 📋 配置说明

### 配置文件（config.yaml）

```yaml
# 模型配置
model:
  path: "yolov8n.pt"
  confidence_threshold: 0.25
  iou_threshold: 0.45
  device: "auto"
  half_precision: false

# 输入配置
input:
  default_image_path: "data/test.jpg"
  camera_index: 0
  input_dir: "data/input"
  supported_formats: [".jpg", ".jpeg", ".png", ".bmp"]

# 输出配置
output:
  output_dir: "data/output"
  save_results: true
  output_format: "jpg"

# 性能配置
performance:
  num_threads: 4
  batch_size: 1

# 显示配置
display:
  show_confidence: true
  show_distance: true
  show_danger_level: true
  show_fps: false

# 危险等级配置
danger:
  danger_threshold: 10.0
  warning_threshold: 20.0
  known_height: 1.6
  focal_length: 700

# 日志配置
logging:
  log_level: "INFO"
  log_file: null
  log_to_console: true
```

### 模型选择建议

| 模型         | 速度   | 精度    | 推荐场景    |
| ---------- | ---- | ----- | ------- |
| yolov8n.pt | ⚡⚡⚡  | ⭐⭐    | 实时性要求高  |
| yolov8s.pt | ⚡⚡   | ⭐⭐⭐   | 平衡性能和精度 |
| yolov8m.pt | ⚡    | ⭐⭐⭐⭐  | 更高精度    |
| yolov8l.pt | 🐌   | ⭐⭐⭐⭐⭐ | 离线处理    |
| yolov8x.pt | 🐌🐌 | ⭐⭐⭐⭐⭐ | 最高精度    |

***

## 📊 多模型对比测试

### 使用方式

```python
from model_comparison import ModelComparison

# 初始化对比器
models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
comparison = ModelComparison(models, conf_threshold=0.25)

# 单图像对比
results = comparison.compare_on_image("test.jpg")

# 批量对比
batch_results = comparison.compare_on_batch("images_folder")

# 基准测试
benchmark = comparison.benchmark_models("test.jpg", runs=10)

# 生成报告
print(comparison.get_comparison_summary())
comparison.export_to_json("report.json")
```

### 对比报告示例

```
多模型对比测试报告
========================================

模型 | 检测总数 | 每图平均 | 平均耗时(ms)
-----|---------|---------|-------------
yolov8n | 156 | 5.2 | 23.5
yolov8s | 168 | 5.6 | 45.2
yolov8m | 172 | 5.7 | 89.1
```

***

## 📈 统计分析

### 使用方式

```python
from stats_analyzer import DetectionStatsAnalyzer

analyzer = DetectionStatsAnalyzer()
print(analyzer.generate_report())
analyzer.save_report()
analyzer.export_to_json()
```

### 统计报告内容

- 类别分布统计
- 置信度分布分析
- 检测数量统计
- 危险目标统计

***

## 💾 结果保存

支持的保存格式：

- 🖼️ **图像文件** - 带检测框标注的图像
- 📄 **JSON数据** - 检测数据详细信息
- 📊 **CSV报告** - 便于Excel分析
- 📋 **批量结果** - 汇总所有检测结果

***

## 📝 测试脚本

| 脚本                         | 功能       |
| -------------------------- | -------- |
| `test_detection.py`        | 静态图像检测测试 |
| `test_camera.py`           | 摄像头检测测试  |
| `test_model_comparison.py` | 多模型对比测试  |
| `test_cli_args.py`         | 命令行参数测试  |
| `test_config.py`           | 配置文件测试   |
| `test_engine.py`           | 检测引擎测试   |
| `test_logger.py`           | 日志系统测试   |
| `test_result_saver.py`     | 结果保存测试   |

***

## 🛠️ 故障排除

### 问题 1：无法找到 ultralytics 模块

```bash
pip install ultralytics --upgrade
```

### 问题 2：摄像头无法打开

- 检查摄像头是否被其他程序占用
- 修改 `config.yaml` 中的 `camera_index`
- 确认系统是否正确识别摄像头

### 问题 3：模型下载失败

- 检查网络连接
- 手动下载模型文件放到项目根目录

### 问题 4：CUDA 内存不足

- 使用更小的模型（如 yolov8n.pt）
- 切换到 CPU 模式
- 降低批量大小

***

## 📚 相关资源

- [Ultralytics YOLOv8 官方文档](https://docs.ultralytics.com/)
- [YOLOv8 GitHub 仓库](https://github.com/ultralytics/ultralytics)
- [OpenCV 官方文档](https://docs.opencv.org/)

***

## 📄 许可证

本项目仅供学习和研究使用。
