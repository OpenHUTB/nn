# YOLO11n 车辆计数系统

这是一个基于 YOLO11n 模型的实时车辆检测、追踪和计数系统。项目使用 Supervision 库进行可视化标注、目标追踪和检测平滑处理，构建了一个健壮的车辆监控系统。

## 项目特点

- **YOLO11n 模型**：使用轻量级 YOLO11n 模型进行高精度实时车辆检测，能够有效区分汽车、摩托车、公交车和卡车等多种车辆类型
- **Supervision 库**：用于标注视频帧、可视化边界框、追踪目标，支持 ByteTrack 高精度目标追踪和 DetectionsSmoother 检测平滑
- **中文界面**：所有注释和文档均为中文，便于理解和维护
- **模块化设计**：路径配置集中管理，支持命令行参数，使用灵活

## 工作流程

1. **目标检测**：使用 YOLO11n 模型检测视频帧中的车辆，输出边界框和分类结果
2. **目标追踪**：使用 ByteTrack 算法跨帧追踪车辆，确保每个车辆只被计数一次
3. **可视化标注**：使用 Supervision 库标注边界框、标签和轨迹线
4. **车辆计数**：基于车辆穿越预设边界线进行计数，实时显示结果

## 项目结构

```
YOLO11n_Vehicle_Counter/
├── main.py                           # 项目入口文件，支持命令行参数
├── scripts/
│   └── yolo_vehicle_counter.py      # 主要车辆计数逻辑
├── models/                          # 存放模型文件
│   └── yolo11n.pt                   # YOLO11n 模型权重
├── dataset/                         # 存放输入视频
│   └── sample.mp4                   # 示例视频
├── res/                             # 存放输出结果
│   └── sample_res.mp4              # 处理结果
├── requirements.txt                 # 依赖包列表
└── README.md                        # 项目说明文档
```

## 安装

1. 安装依赖包：
    ```bash
    pip install -r requirements.txt
    ```

2. **创建项目目录结构**：
    项目使用 `.gitignore` 排除了大文件目录，因此在运行前需要手动创建以下目录：
    ```bash
    mkdir -p models dataset res
    ```

3. 下载模型文件：

   **方法一：从官方源下载**
   - 访问 [YOLO11官方文档](https://docs.ultralytics.com/zh/models/yolo11/) 下载YOLO11n模型
   - 或运行命令下载：
     ```bash
     yolo download model=yolo11n.pt
     ```

   **方法二：从Google Drive下载**
   - 访问 [Google Drive链接](https://drive.google.com/drive/folders/10LTBv6ae3D-Tifn__pSU7krhtJOIl_So?usp=drive_link)
   - 下载 `yolo11n.pt` 模型文件
   - 放置到 `models/` 目录下

4. 准备测试视频：
   - 从上述Google Drive链接中下载示例视频 `sample.mp4`
   - 放置到 `dataset/` 目录下
   - 或准备你自己的视频文件

> **注意**：为了减小项目体积，模型文件和视频文件已从版本控制中排除。你需要手动创建目录并下载所需文件。大型文件存储在Google Drive中，便于分发和更新。

## 使用方法

### 方式一：使用 main.py 命令行入口

```bash
# 使用默认配置运行
python main.py

# 指定自定义路径
python main.py --model models/custom_model.pt --input videos/test.mp4 --output results/output.mp4

# 使用相对路径
python main.py --input ../videos/cars.mp4 --output ../results/cars_counted.mp4

# 查看帮助信息
python main.py --help
```

### 方式二：直接运行脚本

```bash
# 修改 scripts/yolo_vehicle_counter.py 中的路径配置后运行
cd scripts
python yolo_vehicle_counter.py
```

## 配置文件说明

在 `scripts/yolo_vehicle_counter.py` 中，路径配置位于文件开头：

```python
# ==================== 配置路径 ====================
MODEL_PATH = "../models/yolo11n.pt"          # 模型文件路径
INPUT_VIDEO_PATH = "../dataset/sample.mp4"   # 输入视频文件路径
OUTPUT_VIDEO_PATH = "../res/sample_res.mp4"  # 输出视频文件路径
# ==================================================
```

## 功能说明

- **支持车辆类型**：汽车(car)、摩托车(motorbike)、公交车(bus)、卡车(truck)
- **检测置信度**：默认阈值为 0.5，可根据需要调整
- **追踪算法**：ByteTrack，支持高帧率视频
- **计数逻辑**：基于穿越水平线的车辆计数
- **可视化**：
  - 圆角矩形边界框
  - 车辆标签（包含追踪ID和类别）
  - 运动轨迹
  - 半透明覆盖区域
  - 实时计数显示

## 运行效果

### 示例视频
你可以从 [Google Drive链接](https://drive.google.com/drive/folders/10LTBv6ae3D-Tifn__pSU7krhtJOIl_So?usp=drive_link) 下载示例视频进行测试：

1. **`sample.mp4`** - 基础测试视频，展示车辆检测和计数效果
2. **`sample_res.mp4`** - 运行后的测试视频结果。

### 可视化效果
运行程序后，你将看到：
- 实时视频处理窗口
- 车辆边界框和标签（包含车辆ID和类型）
- 车辆运动轨迹线
- 实时车辆计数显示
- 计数线和感兴趣区域(ROI)可视化
- 处理后的视频保存到指定路径

### 支持的车辆类型
- 🚗 汽车 (car)
- 🏍️ 摩托车 (motorbike)
- 🚌 公交车 (bus)
- 🚛 卡车 (truck)

### 交互控制
- 按 'p' 键可以暂停/继续视频播放
- 实时显示处理帧率和计数结果

## 注意事项

- 按 'p' 键可以暂停/继续视频播放
- 确保输入视频清晰，车辆目标可见
- 可根据实际场景调整计数线位置（`limits` 参数）
- 建议使用 GPU 加速以提高处理速度

## 依赖包

- ultralytics (YOLO 模型)
- supervision (标注和追踪)
- opencv-python (视频处理)
- numpy (数值计算)

---

*项目持续更新中，欢迎提出建议和反馈*
