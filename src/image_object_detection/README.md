# YOLOv8 图像目标检测系统

本项目基于 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) 构建，提供一个轻量级、模块化的目标检测工具，支持：

- 🖼️ **静态图像检测**
- 📹 **实时摄像头检测**
- 📦 **批量图像检测**

适用于教学演示、快速验证模型效果或嵌入到边缘设备中。

---

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

**方式一：使用 requirements.txt（推荐）**

```bash
pip install -r requirements.txt
```

**方式二：手动安装**

```bash
pip install ultralytics>=8.0.0
pip install opencv-python>=4.5.0
pip install torch>=2.0.0
pip install matplotlib>=3.3.0
pip install numpy>=1.21.0
pip install pillow>=8.0.0
pip install pyyaml>=5.4.0
pip install psutil>=5.8.0
```

#### 3. 验证安装

运行测试脚本验证环境：

```bash
python test_detection.py
```

---

## 📁 项目目录结构

```
image_object_detection/
│
├── main.py                     # 程序入口文件
├── requirements.txt            # Python 依赖列表
├── config.py                   # 全局配置管理
├── config.yaml                 # 配置文件（YAML格式）
├── detection_engine.py         # YOLO 模型封装核心
├── ui_handler.py               # 用户交互逻辑
├── camera_detector.py          # 实时摄像头检测器
├── batch_detector.py           # 批量图像检测器
├── model_manager.py            # 模型热切换管理器
│
├── test_detection.py           # 静态图像检测测试脚本
├── test_camera.py              # 摄像头检测测试脚本
│
├── data/                       # 测试数据目录（自动创建）
│   ├── test.jpg                # 测试图像
│   └── result.jpg              # 检测结果图像
│
└── README.md                   # 本说明文件
```

---

## 🚀 快速开始

### 方式一：交互式菜单运行（推荐）

直接运行主程序，进入交互式菜单：

```bash
python main.py
```

将看到如下菜单：

```
=== YOLOv8 目标检测系统 ===
1. 静态图像检测
2. 实时摄像头检测
3. 批量图像检测
4. 切换模型
5. 退出程序
```

选择相应的数字即可使用对应功能。

### 方式二：命令行直接运行

```bash
python main.py
```

---

## 📖 使用教程

### 1. 静态图像检测

在交互式菜单中选择 `1`，然后：
- 可以使用默认测试图像
- 或者输入自定义图像路径

检测结果会自动保存到 `data/result.jpg`。

### 2. 实时摄像头检测

在交互式菜单中选择 `2`：
- 程序会打开摄像头窗口
- 按 `q` 键退出检测
- 实时显示检测框和标签
- 显示 FPS 性能统计

### 3. 批量图像检测

在交互式菜单中选择 `3`：
- 输入包含图像的目录路径
- 程序会自动检测所有图像
- 检测结果保存到同目录下

### 4. 切换模型

在交互式菜单中选择 `4`：
- 支持切换不同尺寸的 YOLOv8 模型
- 可选模型：yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
- 模型会自动下载（首次使用时）

---

## ⚙️ 配置说明

### 配置文件（config.yaml）

所有配置项位于 `config.yaml`，可按需修改：

```yaml
model_path: "yolov8n.pt"          # 模型文件路径（支持自动下载）
confidence_threshold: 0.3         # 检测置信度阈值
camera_index: 0                   # 摄像头设备索引
output_interval: 1.0              # FPS 输出间隔（秒）
default_image_path: "data/test.jpg"  # 默认测试图像路径
```

### 配置参数详解

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_path` | `"yolov8n.pt"` | 模型文件路径（nano/small/medium/large/xlarge） |
| `confidence_threshold` | `0.3` | 检测置信度阈值，值越高检测越严格 |
| `camera_index` | `0` | 摄像头设备索引（通常 0 为主摄像头） |
| `output_interval` | `1.0` | FPS 输出间隔（秒） |
| `default_image_path` | `"data/test.jpg"` | 默认测试图像路径 |

### 模型选择建议

| 模型 | 速度 | 精度 | 推荐场景 |
|------|------|------|----------|
| yolov8n.pt | ⚡⚡⚡ | ⭐⭐ | 实时性要求高的场景 |
| yolov8s.pt | ⚡⚡ | ⭐⭐⭐ | 平衡性能和精度 |
| yolov8m.pt | ⚡ | ⭐⭐⭐⭐ | 追求更高精度 |
| yolov8l.pt | 🐌 | ⭐⭐⭐⭐⭐ | 离线处理，追求最佳精度 |
| yolov8x.pt | 🐌🐌 | ⭐⭐⭐⭐⭐ | 最高精度，离线批量处理 |

---

## 📝 测试脚本

项目提供了两个测试脚本：

### test_detection.py

测试静态图像检测功能：

```bash
python test_detection.py
```

该脚本会：
1. 自动下载测试图像
2. 加载 YOLOv8 模型
3. 执行目标检测
4. 保存检测结果

### test_camera.py

测试摄像头实时检测功能：

```bash
python test_camera.py
```

该脚本会：
1. 加载 YOLOv8 模型
2. 打开摄像头
3. 实时进行目标检测
4. 按 `q` 键退出

---

## 💡 使用示例

### 示例 1：检测单张图像

```bash
python main.py
# 选择 1
# 输入图像路径（或使用默认）
```

### 示例 2：实时摄像头检测

```bash
python main.py
# 选择 2
# 按 q 退出
```

### 示例 3：批量检测目录中的图像

```bash
python main.py
# 选择 3
# 输入图像目录路径
```

---

## ⚠️ 注意事项

- 首次运行时，若未提供模型文件，程序会自动从网络下载（需联网）
- 若在无图形界面环境（如服务器、Docker）运行，请避免使用实时摄像头功能
- Windows 用户请确保路径使用正斜杠 `/` 或原始字符串 `r"..."`，避免转义问题
- 摄像头功能需要系统有可用的摄像头设备
- 检测性能取决于硬件配置，建议使用 GPU 加速（需安装 CUDA 版本的 PyTorch）

---

## 🛠️ 故障排除

### 问题 1：无法找到 ultralytics 模块

**解决方案**：
```bash
pip install ultralytics --upgrade
```

### 问题 2：摄像头无法打开

**解决方案**：
- 检查摄像头是否被其他程序占用
- 尝试修改 `config.yaml` 中的 `camera_index` 为 1 或其他数字
- 确认系统是否正确识别摄像头设备

### 问题 3：模型下载失败

**解决方案**：
- 检查网络连接
- 手动从 [Ultralytics 官网](https://github.com/ultralytics/assets/releases) 下载模型文件
- 将下载的模型文件放到项目根目录

---

## 📚 相关资源

- [Ultralytics YOLOv8 官方文档](https://docs.ultralytics.com/)
- [YOLOv8 GitHub 仓库](https://github.com/ultralytics/ultralytics)
- [OpenCV 官方文档](https://docs.opencv.org/)

---

## 📄 许可证

本项目仅供学习和研究使用。
```

