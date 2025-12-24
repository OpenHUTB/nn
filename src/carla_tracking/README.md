# CARLA 目标检测与跟踪系统

这是一个基于CARLA模拟器的车辆目标检测与跟踪系统，集成了YOLOv5目标检测模型和优化版SORT跟踪算法，能够实时检测和跟踪道路上的车辆，并实现基本的自动驾驶控制功能。

## 功能特点

- 基于YOLOv5模型的实时目标检测
- 优化版SORT跟踪算法，支持多目标持续跟踪
- 深度相机数据融合，提升远距离目标跟踪效果
- 车辆自动控制，包括速度调节和障碍物避障
- NPC车辆管理，可生成多个自动驾驶的NPC车辆
- 实时性能监控面板，展示系统各模块运行效率

## 环境要求

- Python 3.8+
- CARLA Simulator 0.9.13+
- 依赖库：
  - carla
  - opencv-python
  - numpy
  - torch
  - ultralytics
  - scipy

## 安装步骤

1. 安装CARLA模拟器：参考[官方文档](https://carla.readthedocs.io/en/latest/start_quickstart/)

2. 安装依赖包：
```bash
pip install carla opencv-python numpy torch ultralytics scipy
```

3. 下载YOLOv5模型文件并放置在指定路径（默认路径：`D:\yolo\`）

## 使用方法

1. 启动CARLA服务器：
```bash
./CarlaUE4.sh  # Linux
CarlaUE4.exe   # Windows
```

2. 运行主程序：
```bash
python main.py [参数选项]
```

## 主要参数选项

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --model | 选择YOLOv5模型 | yolov5m |
| --host | CARLA服务器地址 | localhost |
| --port | CARLA服务器端口 | 2000 |
| --conf-thres | 检测置信度阈值 | 0.15 |
| --iou-thres | IOU阈值 | 0.4 |
| --use-depth | 使用深度相机数据 | True |
| --show-depth | 显示深度图像 | False |
| --npc-count | NPC车辆数量 | 20 |
| --enable-physics | 启用物理模拟 | True |
| --target-speed | 目标速度(km/h) | 30.0 |
| --manual-control | 手动控制模式 | False |

## 系统模块说明

1. **目标检测模块**：使用YOLOv5模型检测图像中的车辆目标

2. **目标跟踪模块**：基于优化版SORT算法，包含：
   - 卡尔曼滤波器预测目标运动轨迹
   - 匈牙利算法进行目标匹配
   - 动态IOU阈值调整，适应不同距离目标

3. **车辆控制模块**：
   - 基于PID的速度控制
   - 障碍物检测与避障
   - 支持手动控制和自动驾驶模式

4. **NPC管理模块**：
   - 生成指定数量的NPC车辆
   - 控制NPC车辆自动驾驶
   - 动态调整NPC行为

5. **传感器模块**：
   - RGB相机获取环境图像
   - 深度相机获取距离信息
   - 数据预处理与优化

## 性能优化

- 使用半精度浮点数加速模型推理（CUDA设备）
- 图像和深度数据的内存优化处理
- 跟踪算法的计算效率优化
- 多线程数据处理与缓冲队列

## 注意事项

- 确保CARLA服务器在运行程序前已启动
- 首次运行会下载YOLOv5模型，可能需要较长时间
- 高分辨率和大量NPC会增加系统负载，可能需要调整参数以获得流畅体验
- 深度相机数据处理对性能影响较大，可通过`--use-depth`参数关闭

## 界面说明

- 主窗口显示摄像头实时画面，包含目标检测框和跟踪信息
- 左上角性能监控面板显示各模块运行时间和FPS
- 目标框颜色随距离变化：近距离(红色)、中距离(黄色)、远距离(绿色)
- 每个目标框显示类别、置信度、跟踪ID、距离和速度信息