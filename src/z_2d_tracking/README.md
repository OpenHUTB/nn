# CARLA 仿真 2D 跟踪：多算法对比与课程实践

## 1. 项目选题

本项目为机器学习课程大作业，旨在自动驾驶仿真平台 CARLA 中实现 2D 多目标跟踪算法，并对比 SORT、DeepSORT、OC-SORT 等经典跟踪器的性能差异。

## 2. 项目目标

- 在 CARLA 环境中实现车辆、行人的 2D 目标跟踪
- 对比不同多目标跟踪算法的效果与鲁棒性
- 完成环境搭建、代码调试、实验分析
- 有余力时扩展至 3D 目标跟踪，更贴近真实自动驾驶感知

## 3. 项目结构

```text
z_2d_tracking/
├── 2d-detection-frcnn.py      # 主程序：Faster R-CNN 目标检测
├── README.md                  # 项目说明文档
├── requirements.txt           # 依赖包列表
├── utils/                     # 工具模块
│   ├── box_utils.py          # 边界框绘制与可视化
│   ├── encorder.py           # 图像特征编码器
│   ├── projection.py         # 3D到2D投影变换
│   └── world.py              # CARLA世界管理
└── what/                      # 攻击算法模块
    └── attacks/
        └── detection/
            ├── yolo/         # YOLO检测器攻击
            └── yolox/        # YOLOX检测器攻击
```

## 4. 运行环境

| 环境 | 版本要求 |
| --- | --- |
| Python | 3.9+ |
| CARLA | 0.9.14 |
| 操作系统 | Windows 11 |
| GPU | 推荐 NVIDIA GPU（用于深度学习） |

## 5. 主要依赖

**核心依赖：**

- `numpy>=1.21.0` - 数值计算
- `opencv-python>=4.5.0` - 图像处理
- `carla>=0.9.13` - CARLA Python API

**深度学习框架（可选）：**

- `tensorflow>=2.10.0,<2.13.0` - 特征编码
- `torch>=1.9.0` - YOLO检测器

**跟踪算法依赖（可选）：**

- `filterpy>=1.4.5` - 卡尔曼滤波
- `lap>=0.4.0` - 匈牙利算法
- `scipy>=1.7.0` - 距离计算

## 6. 安装与运行

### 安装依赖

```bash
pip install -r requirements.txt
```

### CARLA 安装

1. 下载 CARLA 仿真器：<https://github.com/carla-simulator/carla/releases>
2. 推荐版本：CARLA 0.9.14
3. 添加 CARLA Python API 到 Python 路径：

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/CARLA/PythonAPI/carla/dist/carla-*.egg
```

### 运行项目

```bash
# 启动 CARLA 仿真器
./CarlaUE4.sh  # Linux
# 或 CarlaUE4.exe  # Windows

# 运行主程序
python 2d-detection-frcnn.py
```

## 7. 实现计划

1. 完成 CARLA 环境配置与调试
2. 实现基于真值的 2D 目标检测与跟踪
3. 集成 YOLO 检测器 + SORT/DeepSORT 跟踪器
4. 对比不同算法在不同场景下的性能
5. 结果可视化与分析总结
6. 可选扩展：3D 目标跟踪实现与对比

## 8. 参考与致谢

本项目基于优秀开源项目学习与复现：

- 基础框架参考：[wuhanstudio/2d-carla-tracking](https://github.com/wuhanstudio/2d-carla-tracking)

在此向原作者表示感谢，本项目在此基础上完成课程实践、算法对比与扩展尝试。