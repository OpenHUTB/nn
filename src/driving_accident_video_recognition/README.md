# Driving Accident Video Recognition

基于 YOLOv8 的驾驶事故视频识别工具，支持实时视频流分析和事故检测。

## 功能特性

- 实时视频/摄像头事故检测
- 多语言支持（中文/英文标注）
- 可调节检测置信度阈值
- 事故区域限定（ROI）
- 检测结果统计（事故、人员、车辆数量）
- 识别视频保存

## 项目结构

```
driving_accident_video_recognition/
├── main.py              # 主程序入口
├── config.py            # 配置文件（YOLO模型、检测参数）
├── detector.py          # 事故检测核心模块
├── process.py           # 视频处理模块
├── dependencies.py      # 依赖管理
├── requirements.txt     # Python 依赖
└── utils/              # 工具模块
```

## 依赖

- Python 3.8+
- Ultralytics (YOLOv8)
- OpenCV
- NumPy
- python-dotenv

## 使用方法

```bash
# 安装依赖
pip install -r requirements.txt

# 基本用法
python main.py --source 0

# 指定视频文件
python main.py --source video.mp4

# 设置置信度阈值
python main.py --conf 0.7

# 启用统计功能
python main.py --enable-stats

# 保存识别后的视频
python main.py --save-path output.mp4

# 设置检测区域（相对坐标）
python main.py --roi 0.2,0.3,0.8,0.7
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--source`, `-s` | 检测源（摄像头编号或视频路径） | 0 |
| `--language`, `-l` | 标注语言（zh/en） | zh |
| `--conf`, `-c` | 检测置信度阈值（0-1） | 0.5 |
| `--save-path`, `-sp` | 保存识别视频路径 | 不保存 |
| `--enable-stats`, `-es` | 启用检测统计 | False |
| `--roi`, `-r` | 检测区域（x1,y1,x2,y2） | 全画面 |

## 配置

可以通过环境变量或 `.env` 文件配置：

```bash
YOLO_MODEL_PATH=yolov8n.pt
CONFIDENCE_THRESHOLD=0.5
DETECTION_SOURCE=0
```
