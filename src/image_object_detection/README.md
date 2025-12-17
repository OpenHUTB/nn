#  YOLOv8 图像目标检测系统

本项目基于 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) 构建，提供一个轻量级、模块化的目标检测工具，支持：

- 🖼️ **静态图像检测**
- 📹 **实时摄像头检测**

适用于教学演示、快速验证模型效果或嵌入到边缘设备中。

---

## 📁 项目目录结构
    yolov8-detection-project/
    │
    ├── main.py                     # 程序入口
    ├── requirements.txt            # Python 依赖列表
    ├── README.md                   # 本说明文件
    │
    ├── config.py                   # 全局配置（模型路径、阈值、默认图像等）
    ├── ui_handler.py               # 用户交互逻辑（命令行 + 交互式菜单）
    ├── detection_engine.py         # YOLO 模型封装（加载、推理、抑制日志）
    ├── image_detector.py           # 静态图像检测器
    ├── camera_detector.py          # 实时摄像头检测器
    │
    ├── yolov8n.pt                  # （可选）YOLOv8 默认模型文件（首次运行自动下载）



## ▶️ 运行方式

### 方式一：命令行直接运行（推荐）

```bash
python main.py --image path/to/your/image.jpg
```

### 方式二：交互式菜单（无参数时自动进入）

```bash
python main.py
```

将看到如下菜单：

```
=== YOLO Detection System ===
1. Static Image Detection
2. Live Camera Detection
3. Exit
```

- 选择 **1** 后可使用默认图像或输入自定义路径。
- 选择 **2** 启动摄像头，按 `q` 退出，按 `s` 保存当前帧。

---

##  配置说明

所有配置项位于 `config.py`，可按需修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_path` | `"yolov8n.pt"` | 模型文件路径（支持自动下载） |
| `confidence_threshold` | `0.25` | 检测置信度阈值 |
| `camera_index` | `0` | 摄像头设备索引（通常 0 为主摄像头） |
| `output_interval` | `1.0` | FPS 输出间隔（秒） |
| `default_image_path` | `./data/test.jpg` | 默认测试图像路径 |

---

##  注意事项

- 首次运行时，若未提供 `yolov8n.pt`，程序会自动从网络下载模型（需联网）。
- 若在无图形界面环境（如服务器、Docker）运行，请避免使用 `cv2.imshow`。
- Windows 用户请确保路径使用正斜杠 `/` 或原始字符串 `r"..."`，避免转义问题。
```

