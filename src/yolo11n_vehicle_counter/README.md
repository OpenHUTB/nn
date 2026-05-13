# YOLO11n 车辆计数系统

这是一个基于 YOLO11n 模型的实时车辆检测、追踪和计数系统。项目使用 Supervision 库进行可视化标注、目标追踪和检测平滑处理，构建了一个健壮的车辆监控系统。

## 项目特点

- **YOLO11n 模型**：使用轻量级 YOLO11n 模型进行高精度实时车辆检测，能够有效区分汽车、摩托车、公交车和卡车等多种车辆类型
- **Supervision 库**：用于标注视频帧、可视化边界框、追踪目标，支持 ByteTrack 高精度目标追踪和 DetectionsSmoother 检测平滑
- **卡尔曼滤波**：基于二维卡尔曼滤波器实现速度平滑，减少数据抖动和噪声（CARLA版本）
- **中文界面**：所有注释和文档均为中文，便于理解和维护
- **模块化设计**：路径配置集中管理，支持命令行参数，使用灵活
- **可拖拽区域**：支持可视化调整计数区域位置和大小

## 工作流程

1. **目标检测**：使用 YOLO11n 模型检测视频帧中的车辆，输出边界框和分类结果
2. **目标追踪**：使用 ByteTrack 算法跨帧追踪车辆，确保每个车辆只被计数一次
3. **速度估算**：基于卡尔曼滤波对车辆速度进行平滑估算（CARLA版本）
4. **可视化标注**：使用 Supervision 库标注边界框、标签、轨迹和置信度分布
5. **车辆计数**：基于车辆穿越预设边界线进行计数，实时显示结果和分类统计

## 项目结构

```
YOLO11n_Vehicle_Counter/
├── main.py                           # 项目入口文件，支持命令行参数
├── scripts/
│   ├── yolo_vehicle_counter.py       # 主要车辆计数逻辑
│   ├── yolo_vehicle_counter_carla.py # CARLA专用版本（速度平滑）
│   ├── yolo_vehicle_counter_improved.py # 改进版本（椭圆标注器）
│   ├── yolo_vehicle_counter_updown.py # 上下行计数版本
│   ├── yolo_vehicle_counter_region.py # 区域计数版本
│   ├── draggable_handler.py          # 可拖拽区域处理器
│   ├── keyboard_handler.py           # 键盘事件处理器
│   └── carla_launcher.py            # CARLA启动器
├── config/
│   └── counting_region.json          # 计数区域配置文件
├── models/                           # 存放模型文件
│   └── yolo11n.pt                    # YOLO11n 模型权重
├── dataset/                          # 存放输入视频
│   └── sample.mp4                    # 示例视频
├── res/                              # 存放输出结果
│   └── sample_res.mp4               # 处理结果
├── screenshots/                      # 截图保存目录（自动创建）
├── requirements.txt                  # 依赖包列表
└── README.md                         # 项目说明文档
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
- **检测置信度**：默认阈值为 0.6，可根据需要调整（CARLA版本支持↑/↓键动态调整）
- **追踪算法**：ByteTrack，支持高帧率视频
- **速度估算**：基于卡尔曼滤波的实时速度平滑估算（CARLA版本）
- **计数逻辑**：基于穿越区域中心线的车辆计数，支持双向交通流统计
- **分类统计**：按车辆类型分类统计计数（CARLA版本）
- **精度衡量**：支持与ground truth数据对比，计算检测精度
  - 支持简单数字格式（整个视频的真实车辆总数）
  - 支持分段验证格式（每行一个数字）
  - 实时显示精度指标（ACC、F1分数）
  - 处理结束后输出详细的精度报告
- **可视化**：
  - 圆角矩形边界框
  - 车辆标签（包含追踪ID、类别、速度、轨迹长度）
  - 运动轨迹
  - 半透明覆盖区域
  - 实时计数和精度显示
  - 可拖拽计数区域（编辑模式）
  - 置信度分布可视化（CARLA版本）
- **可拖拽区域**：支持可视化调整计数区域位置和大小
  - 8个控制点（4个角点 + 4个边中点）
  - 自动保存区域配置到 JSON 文件
  - 支持区域尺寸限制和边界检测
- **速度计算**：实时计算并显示车辆速度
  - 支持 km/h 和 m/s 单位切换
  - 速度平滑显示，避免抖动
- **自动区域调整**：根据检测结果动态调整计数区域
- **分车型统计**：分别统计汽车、摩托车、公交车、卡车数量

## 运行效果

### 示例视频
你可以从 [Google Drive链接](https://drive.google.com/drive/folders/10LTBv6ae3D-Tifn__pSU7krhtJOIl_So?usp=drive_link) 下载示例视频进行测试：

1. **`sample.mp4`** - 基础测试视频，展示车辆检测和计数效果
2. **`sample_res.mp4`** - 原始版本运行结果示例
3. **`sample_res_improved.mp4`** - 改进版本运行结果示例

### 运行输出示例
```
视频信息: 1920x1080, 30fps
按 'p' 键暂停，'↑/↓' 键调整置信度
按 'q' 键退出

当前置信度阈值: 0.40
实时计数: 156
上行: 82 | 下行: 74
汽车: 120 | 摩托车: 15 | 公交车: 12 | 卡车: 9
FPS: 45.2
```

### 可视化效果
运行程序后，你将看到：
- 实时视频处理窗口
- 车辆边界框和标签（包含车辆ID、类型、速度、轨迹长度）
- 车辆运动轨迹线
- 实时车辆计数显示（包含分类统计）
- 计数线和感兴趣区域(ROI)可视化
- 置信度分布条形图（CARLA版本）
- 处理后的视频保存到指定路径

### 支持的车辆类型
- 🚗 汽车 (car)
- 🏍️ 摩托车 (motorbike)
- 🚌 公交车 (bus)
- 🚛 卡车 (truck)

### 交互控制

| 按键 | 功能 |
|------|------|
| `d` | 切换编辑模式（可拖拽调整计数区域） |
| `p` | 暂停/继续视频播放 |
| `空格` | 逐帧播放（暂停时） |
| `s` | 保存当前帧截图到 `screenshots/` 目录 |
| `↑` | 增加置信度阈值（仅 CARLA 版本） |
| `↓` | 降低置信度阈值（仅 CARLA 版本） |
| `q` | 退出程序 |

**编辑模式说明**：
- 按 `d` 键进入编辑模式后，计数区域会显示8个绿色控制点
- 鼠标悬停在控制点上会变为黄色高亮
- 拖拽控制点可调整区域位置和大小
- 区域尺寸实时显示在区域中心
- 再次按 `d` 键退出编辑模式，配置自动保存

## 版本特性

### 版本说明
项目包含五个版本，可通过 `main.py` 的 `--version` 参数选择：

1. **原始版本** (`original`) - 基础车辆计数功能
2. **改进版本** (`improved`) - 增强的检测逻辑和可视化效果
3. **上下行计数版本** (`updown`) - 多车道系统，分别统计上下行方向的车辆
4. **CARLA专用版本** (`carla`) - 专门处理CARLA模拟器录制视频的优化版本
5. **区域计数版本** (`region`) - 高级迭代版本，支持多边形区域计数和双向交通流统计

### 改进版本主要特性

**🚀 增强的检测和计数逻辑**
- 优化车辆检测算法，提高检测准确率
- 改进计数逻辑，减少误检和漏检
- 支持更复杂的交通场景

**🎨 优化的可视化标注工具**
- **椭圆标注器** (`EllipseAnnotator`) - 替代传统的矩形边界框，提供更美观的车辆标注效果
- **增强的标签系统** - 优化标签位置和样式，提升可视化体验
- **改进的轨迹追踪** - 使用顶部中心位置绘制轨迹，更直观显示车辆运动路径

**📐 多边形感兴趣区域 (Polygon ROI)**
- 使用多边形掩码替代矩形ROI，支持更复杂的监控区域设置
- 可自定义多边形顶点，适应不同的道路布局
- 提高复杂背景下的检测准确性

**🔧 参数优化**
- 调整计数线位置和覆盖区域，适应不同分辨率的视频
- 优化覆盖区域透明度，平衡可视化和视频内容显示
- 改进检测置信度阈值，提高计数精度

### 上下行计数版本特性

**🚦 多车道车辆计数系统**
- 在之前版本的基础上增加了多车道车辆计数功能
- 分别统计上行（up）和下行（down）方向的车辆
- 为上下行交通流定义了不同的计数线

### CARLA专用版本特性

**🎮 CARLA模拟器视频优化**
- 专门针对CARLA模拟器录制的视频进行优化
- 针对模拟环境的特殊视角和车辆行为调整检测参数
- 优化计数线位置，适配CARLA视频的透视效果

**🔧 技术优化**
- **红线位置调整**：从y=400降低到y=500，更适合CARLA视频视角
- **ROI区域优化**：调整感兴趣区域，减少天空干扰，专注道路区域
- **置信度阈值**：从0.5降低到0.4，适应模拟环境检测
- **计数区域**：扩大垂直范围，更容易捕捉CARLA车辆
- **卡尔曼滤波**：基于二维卡尔曼滤波器实现速度平滑

**📊 使用场景**
- 处理CARLA模拟器录制的交通流视频
- 测试车辆检测算法在模拟环境中的表现
- 生成训练数据和验证数据

### 区域计数版本特性

**🚀 高级检测优化**
- 这是对原始版本的进一步优化迭代，提供更平滑的检测体验
- **检测结果平滑处理**：避免车辆位置突然变化，提供更稳定的追踪效果
- **双向交通流计数**：同时统计上行和下行方向的车辆流量
- **多边形区域计数**：支持自定义多边形区域，适应复杂道路布局

**🎨 定制化视觉增强**
- **半透明覆盖区域**：使用半透明多边形覆盖计数区域，清晰标识监控范围
- **轨迹注解**：显示车辆运动路径，便于分析交通流模式
- **分区计数显示**：实时显示总车辆数、上行车辆数和下行车辆数
- **优化的标注样式**：改进的边界框和标签显示，提升可视化体验

**🔧 技术特性**
- **多边形ROI支持**：可自定义多边形顶点，精确控制监控区域
- **动态区域缩放**：根据视频分辨率自动缩放区域坐标
- **置信度过滤**：0.5置信度阈值，平衡检测精度和召回率
- **底部中心计数**：基于车辆底部中心点进行区域检测，提高计数准确性

**📊 实时显示上下行计数**
- 实时显示总车辆数（COUNTS）
- 单独显示上行车辆数（UP）
- 单独显示下行车辆数（DOWN）
- 支持双向交通流分别监控和统计

**🔧 分区计数逻辑**
- 使用分区限制（partition_limit）区分上下行区域
- 左侧区域（0-550px）统计上行车辆
- 右侧区域（550px-1280px）统计下行车辆
- 支持复杂道路场景的精确计数

### 使用示例

```bash
# 运行原始版本(默认)
python main.py --version original

# 运行改进版本
python main.py --version improved

# 运行上下行计数版本
python main.py --version updown

# 运行CARLA专用版本
python main.py --version carla

# 运行区域计数版本
python main.py --version region

# 运行原始版本并指定ground truth文件进行精度衡量
python main.py --version original --ground-truth dataset/ground_truth/ground_truth.txt

# 运行改进版本并指定输出到improved目录
python main.py --version improved --output res/improved/result.mp4

# 运行上下行计数版本
python main.py --version updown --input dataset/sample_updown.mp4 --output res/improved/sample_updown_res.mp4

# 运行CARLA专用版本处理CARLA录制视频
python main.py --version carla --input dataset/carla_recording.mp4 --output res/carla/carla_counted.mp4

# 运行区域计数版本处理自定义视频
python main.py --version region --input dataset/highway.mp4 --output res/region/highway_counted.mp4
```

### 技术改进详解

1. **标注优化**: 椭圆标注器相比矩形标注器，在视觉上更加美观，特别是在车辆角度变化较大时效果更佳
2. **ROI优化**: 多边形ROI能够更精确地定义监控区域，排除无关区域的干扰
3. **轨迹优化**: 顶部中心轨迹显示更符合车辆运动特性，便于观察车辆行驶路径
4. **参数调优**: 各项参数经过优化，在保持检测速度的同时提高计数准确率
5. **多方向计数**: 支持上下行分别计数，满足不同交通监控需求
6. **可拖拽区域**: 可视化调整计数区域，提升用户交互体验
7. **卡尔曼滤波**: 速度平滑估算，减少数据抖动

---

*项目持续更新中，欢迎提出使用反馈和建议*

*改进版本持续优化中，欢迎提出使用反馈和建议*

## 注意事项

- 按 `d` 键进入编辑模式，可拖拽调整计数区域
- 按 `p` 键可以暂停/继续视频播放
- 按 `s` 键保存当前帧截图到 `screenshots/` 目录
- 按 `空格` 键在暂停时逐帧播放
- 按 `q` 键退出程序
- 确保输入视频清晰，车辆目标可见
- 计数区域配置会自动保存到 `config/counting_region.json`
- 建议使用 GPU 加速以提高处理速度

## 新增文件说明

### `scripts/draggable_handler.py`
可拖拽区域处理器，提供以下功能：
- 支持8个控制点（4个角点 + 4个边中点）拖拽调整
- 配置保存和加载到 JSON 文件
- 编辑模式下的视觉反馈（不同颜色表示拖拽/悬停状态）
- 区域尺寸限制和边界检测

### `scripts/keyboard_handler.py`
键盘事件处理器，提供以下功能：
- 编辑模式切换（`d` 键）
- 暂停/继续播放（`p` 键）
- 逐帧播放（空格）
- 截图保存（`s` 键）
- 程序退出（`q` 键）

### `config/counting_region.json`
计数区域配置文件，存储：
- 区域坐标 [left, top, right, bottom]
- 控制点半径
- 区域最小尺寸

## 依赖包

- ultralytics (YOLO 模型)
- supervision (标注和追踪)
- opencv-python (视频处理)
- numpy (数值计算)

---

*项目持续更新中，欢迎提出建议和反馈*