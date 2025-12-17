# CARLA_YOLO_ObjectTrackingTracking

## 项目概述
本项目实现了基于YOLOv5算法和CARLA仿真器的自动驾驶目标检测与跟踪功能。通过CARLA仿真环境生成动态交通场景，利用YOLOv5模型实时检测车辆、摩托车、公交车、卡车等交通参与者，结合内置优化版SORT跟踪算法实现目标的连续追踪，并通过OpenCV实时可视化检测与跟踪结果。

### 核心功能：
- 基于YOLOv5模型（支持yolov5s、yolov5m、yolov5x）进行实时目标检测，默认置信度阈值0.25
- 集成优化版SORT跟踪算法（内置实现，无需额外依赖），通过卡尔曼滤波和历史位置平滑处理，实现对检测目标的稳定跟踪，支持多目标ID持续标注
- 与CARLA仿真器深度集成，自动生成主车辆和NPC交通流，挂载RGB相机采集实时图像（分辨率640×480）
- 支持同步模式运行，保证检测帧率与仿真环境一致（默认0.05s/帧，20FPS）
- 实时可视化检测/跟踪结果，显示目标类别、置信度及跟踪ID，不同类别目标使用差异化颜色标注

## 安装步骤

### 前置条件：
- Python 3.7+（推荐3.7，与CARLA兼容性更佳）
- CARLA仿真器（支持0.9.x系列版本，版本需与Python API匹配）
- NVIDIA显卡（支持CUDA 11.3+，可选，用于加速模型推理）

### 操作步骤：

1. 下载并安装CARLA仿真器：
```plaintext
https://github.com/carla-simulator/carla/releases
```
选择适合操作系统的版本，解压至任意路径（如D:\CARLA）

2. 克隆或下载本项目代码：
```bash
git clone <项目仓库地址>
cd CARLA_YOLO_ObjectTracking
```

3. 安装Python依赖库：
```bash
# 安装PyTorch（支持CUDA加速）
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113

# 安装YOLOv5依赖
pip install ultralytics

# 安装其他依赖
pip install numpy opencv-python scipy carla

# 若使用CPU版本PyTorch
pip install torch torchvision
```

4. 安装CARLA Python API：
```bash
# 进入CARLA安装目录下的PythonAPI路径
cd D:\CARLA\PythonAPI\carla\dist
# 安装对应版本的egg文件（需与Python版本匹配）
easy_install carla-<版本号>-py3.7-<平台>.egg
```

5. 下载YOLOv5预训练模型：
将模型文件（yolov5s.pt/yolov5m.pt/yolov5x.pt）放置在指定路径（默认路径为`D:\yolo\`，可在代码`load_detection_model`函数中修改）

## 使用方法

1. 启动CARLA仿真器：
```bash
cd D:\CARLA
# Windows系统
CarlaUE4.exe
# Linux系统
./CarlaUE4.sh
```
等待地图加载完成（显示3D城市场景）

2. 运行主程序：
```bash
# 基础运行（默认yolov5m模型+SORT跟踪器）
python main.py

# 可选参数配置
python main.py --model yolov5x --tracker sort --host localhost --port 2000 --conf-thres 0.25 --iou-thres 0.45
```

3. 操作指令：
- 按`q`键：退出程序
- 程序运行时会自动生成主车辆、NPC车辆及相机，实时显示检测跟踪结果

## 功能说明

1. 目标检测：
- 支持检测类别：汽车（car）、摩托车（motorcycle）、公交车（bus）、卡车（truck）（对应COCO数据集ID：2、3、5、7）
- 模型选择：yolov5s（轻量快速）、yolov5m（平衡精度与速度）、yolov5x（高精度，适合对精度要求高的场景）
- 检测优化：
  - 采用多帧投票机制（保留至少1帧出现的目标），提高小目标检测稳定性
  - 过滤过小检测框（宽度和高度均大于8像素），减少误检
  - 支持置信度阈值（--conf-thres）和NMS IOU阈值（--iou-thres）配置

2. 目标跟踪：
- 采用优化版SORT跟踪算法，核心改进包括：
  - 卡尔曼滤波器参数优化，适配车辆运动特性
  - 引入历史位置平滑处理，减少跟踪抖动
  - 提高IOU匹配阈值，增强跟踪稳定性
  - 过滤过小检测框，减少误跟踪
- 跟踪核心参数：max_age（最大消失帧数，默认5）、min_hits（最小命中数，默认3）、iou_threshold（IOU匹配阈值，默认0.4）
- 跟踪结果包含持续的目标ID，支持目标短暂消失后重新出现的ID匹配

3. CARLA交互：
- 自动清理仿真环境中的残留车辆和静态车辆，保证场景纯净
- 主车辆开启自动驾驶模式，相机优化挂载于主车辆前方（x=2.0m，z=1.5m），更适合前方车辆检测
- 自动生成NPC车辆并开启自动驾驶，构建动态交通场景（默认生成10辆）
- spectator视角自动跟随主车辆（后上方5m，俯角30°），方便观察全局场景
- 同步模式运行，固定帧率0.05s/帧（20FPS），保证检测稳定性

## 项目结构
- main.py：主程序脚本，包含以下核心组件：
  - KalmanFilter：卡尔曼滤波器实现，用于目标运动状态预测与更新
  - Track：跟踪目标实体类，维护单目标的边界框、ID、生命周期及历史位置等信息
  - Sort：SORT跟踪器核心实现，处理多目标匹配与跟踪状态管理
  - 工具函数：包含边界框绘制（支持类别差异化颜色）、相机投影矩阵构建、CARLA环境清理等功能
  - CARLA交互模块：负责服务器连接、车辆生成、相机配置及同步控制
  - 检测与跟踪主逻辑：集成YOLOv5检测与SORT跟踪，实现端到端流程
- YOLOv5预训练模型（yolov5s.pt/yolov5m.pt/yolov5x.pt）：需放置在指定路径

## 常见问题

### CARLA连接失败：
- 确保CARLA仿真器已启动，且端口与脚本参数一致（默认2000）
- 检查CARLA版本与Python API版本是否匹配
- 关闭防火墙或添加端口例外规则

### 模型加载错误：
- 确认模型文件已下载并放在正确路径（默认`D:\yolo\`）
- 检查模型文件名与代码中`model_paths`字典配置一致
- 网络问题导致模型自动下载失败时，可手动下载并放置到指定路径

### 性能问题：
- 检测速度慢：可切换为yolov5s模型，或降低相机分辨率（修改`camera_bp`的`image_size_x`和`image_size_y`参数）
- 仿真卡顿：减少NPC车辆数量（修改`spawn_npcs`函数的`count`参数）
- 内存占用高：关闭不必要的后台程序，或降低模型复杂度

### 可视化窗口无响应：
- 确保OpenCV版本兼容（推荐4.5.x系列）
- 检查是否存在未处理的异常导致主循环中断
- 尝试重新启动CARLA仿真器和程序

## 参考文档
- [CARLA官方文档](https://carla.readthedocs.io/)
- [YOLOv5官方文档](https://docs.ultralytics.com/yolov5/)
- [SORT跟踪算法论文](https://arxiv.org/abs/1602.00763)

## 许可协议
本项目基于MIT许可协议开源，详情参见LICENSE文件。