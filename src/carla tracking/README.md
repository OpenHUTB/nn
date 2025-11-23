# Object Tracking in CARLA

本项目基于CARLA模拟器实现了2D和3D目标检测与跟踪功能，支持多种检测模型（YOLO、SSD、Faster RCNN等）和跟踪算法（SORT、Deep SORT、Strong SORT、OC SORT等），可用于自动驾驶场景下的目标感知研究与测试。


## 环境要求

项目依赖以下Python库，可通过`requirements.txt`安装：
```
whitebox-adversarial-toolbox
filterpy==1.4.5
scikit-image==0.17.2
lap==0.4.0
carla==0.9.14
```

安装命令：
```bash
pip install -r requirements.txt
```

此外，需要安装CARLA模拟器（0.9.14版本），并确保模拟器服务在本地`localhost:2000`端口运行。


## 功能说明

### 1. 2D目标检测

支持多种2D目标检测模型，可实时检测CARLA场景中的车辆（汽车、公交车、卡车等）。

| 检测模型 | 运行命令 |
|----------|----------|
| YOLOv4 | `python 2d-detection-yolo.py` |
| SSD | `python 2d-detection-ssd.py` |
| Faster RCNN | `python 2d-detection-frcnn.py` |

示例效果（YOLOv4检测）：
![](docs/2d-detection-yolo.gif)


### 2. 2D目标跟踪

结合检测模型与跟踪算法，实现对动态目标的连续跟踪，输出目标ID和边界框。

支持的检测+跟踪组合及运行命令：

| 检测源 | 跟踪算法 | 运行命令 |
|--------|----------|----------|
| 真值（Ground Truth） | SORT | `python 2d-tracking-gt-sort.py` |
| 真值（Ground Truth） | Deep SORT | `python 2d-tracking-gt-deep-sort.py` |
| 真值（Ground Truth） | Strong SORT | `python 2d-tracking-gt-strong-sort.py` |
| 真值（Ground Truth） | OC SORT | `python 2d-tracking-gt-oc-sort.py` |
| YOLO检测 | SORT | `python 2d-tracking-yolo-sort.py` |
| YOLO检测 | Deep SORT | `python 2d-tracking-yolo-deep-sort.py` |
| YOLO检测 | Strong SORT | `python 2d-tracking-yolo-strong-sort.py` |
| YOLO检测 | OC SORT | `python 2d-tracking-yolo-oc-sort.py` |

示例效果（真值+SORT跟踪）：
![](docs/2d-tracking-gt-sort.gif)


### 3. 3D目标跟踪

基于3D真值检测信息，结合SORT算法实现3D目标跟踪。

| 功能 | 运行命令 |
|------|----------|
| 3D真值检测 | `python 3d-detection-gt.py` |
| 3D真值+SORT跟踪 | `python 3d-tracking-gt-sort.py` |

示例效果（3D真值+SORT跟踪）：
![](docs/3d-tracking-gt-sort.gif)


## 核心模块说明

- **检测模块**：集成YOLOv4、SSD、Faster RCNN等模型，实现目标检测与边界框输出。其中：
  - `2d-detection-yolo.py`：YOLOv4模型的2D检测实现，支持车辆、公交车、卡车等目标检测
  - `2d-detection-frcnn.py`：Faster RCNN模型的2D检测实现，基于PyTorch框架
- **跟踪模块**：包含SORT、Deep SORT、Strong SORT等跟踪算法，其中：
  - `deep_sort/`：Deep SORT算法实现，包含卡尔曼滤波器（`kalman_filter.py`）、跟踪器（`track.py`）等核心组件，通过特征提取器进行目标关联
  - `strong_sort/`：Strong SORT算法实现，在Deep SORT基础上优化了卡尔曼滤波（引入置信度加权，`kalman_filter.py`中`project`方法）和特征更新策略（EMA平滑，`track.py`中`update`方法）
- **CARLA交互模块**：`utils/world.py`负责CARLA世界初始化、车辆生成、传感器配置及场景清理等操作
- **工具函数**：
  - `utils/projection.py`：实现3D坐标到2D图像的投影转换，包含相机内参矩阵构建、空间坐标转换等功能
  - `utils/encoder.py`：提供目标特征提取功能，用于Deep SORT/Strong SORT中的外观特征匹配
  - `utils/box_utils.py`：负责边界框绘制、坐标转换等辅助功能


## 使用说明

1. 启动CARLA模拟器：
   ```bash
   ./CarlaUE4.sh
   ```

2. 运行对应功能的脚本（如2D YOLO+Deep SORT跟踪）：
   ```bash
   python 2d-tracking-yolo-deep-sort.py
   ```

3. 按`q`键退出程序。


## 注意事项

- Windows系统需在相机配置中设置`image_size_x`和`image_size_y`为640以避免显示问题（代码中已默认配置）
- 首次运行时会自动下载所需的模型权重文件（如YOLOv4、Faster RCNN权重）
- 可通过调整脚本中的参数（如`range(50)`控制NPC车辆数量、检测阈值等）修改场景配置
- 跟踪算法中，Strong SORT相比Deep SORT引入了置信度加权的卡尔曼滤波和EMA特征平滑，提升了复杂场景下的跟踪稳定性