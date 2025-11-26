# 自动驾驶系统（基于 CARLA 与深度学习）

## 项目概述
本项目构建了基于 CARLA 仿真平台的自动驾驶系统，通过深度学习技术提升车辆感知能力，集成多模态传感器实现全面环境认知，提供模块化、可扩展的代码库支持自动驾驶算法研发。借助 CARLA 的高保真仿真能力，模拟真实世界驾驶场景与挑战，为自动驾驶技术的学习、研究与开发提供可靠环境。

## 环境准备

### 依赖库安装
```bash
# 建议使用Python 3.7及以上版本（推荐虚拟环境）
pip install carla
pip install numpy opencv-python keras tensorflow pygame matplotlib
```

#### 依赖说明：
- carla：自动驾驶高保真仿真平台核心库
- python 3.7+：项目开发与运行的 Python 版本
- opencv-python：图像数据处理与可视化
- keras：深度学习语义分割模型构建
- tensorflow：深度学习模型训练与优化
- pygame：手动控制车辆功能支持
- numpy：数值计算基础支持
- matplotlib：数据可视化工具

### 开发环境配置
1. 下载并安装[CARLA 官方发行版](https://github.com/carla-simulator/carla/releases)（推荐最新稳定版本）
2. 安装[VSCode](https://code.visualstudio.com/)并配置 Python 3.7+ 解释器
3. 推荐插件：Python、Pylance、Code Runner（提升开发效率）

## 项目结构

| 文件名             | 功能描述                                                     |
|--------------------|--------------------------------------------------------------|
| `main.py`          | 核心程序入口，负责 CARLA 客户端连接、世界初始化与主循环控制   |
| `perception.py`    | 感知模块，基于深度学习实现语义分割与环境要素识别               |
| `sensor_manager.py`| 传感器管理模块，处理 RGBA 摄像头、LiDAR 等多模态数据采集与同步 |
| `model_trainer.py` | 模型训练工具，提供语义分割 CNN 模型的训练、验证与优化流程     |
| `utils.py`         | 通用工具函数库，包含数据转换、可视化与性能评估等辅助功能       |
| `config.yaml`      | 配置文件，存储仿真参数（传感器类型、模型参数、仿真帧率等）     |
| `README.md`        | 项目说明文档                                                   |

## 核心功能

### 1. 高保真仿真环境
- 基于 CARLA 构建多样化驾驶场景，支持天气（晴、雨、雾）、时间（昼、夜）等环境动态调整
- 模拟多车辆与交通参与者，还原复杂交通流场景
- 支持 CARLA 服务器自动连接、断开重连与仿真状态实时监控

### 2. 多模态感知系统
- 集成 RGBA 摄像头（色彩纹理信息）、LiDAR（深度距离信息）等多类传感器
- 基于深度学习实现实时语义分割，精准识别道路、车辆、行人、交通标志等核心要素
- 优化传感器数据与车辆状态的时间戳同步，提升感知准确性

### 3. 深度学习语义分割
- 针对自动驾驶场景优化的 CNN 网络架构，兼顾分割精度与实时性
- 提供完整训练流程：数据加载、模型训练、损失计算与权重优化
- 支持交并比（IoU）等核心指标评估，量化模型性能

### 4. 数据收集与可视化
- 支持批量采集传感器数据（图像、点云）与对应标注，用于模型训练
- 实时展示车辆动态、传感器原始数据与语义分割结果，便于直观分析系统性能

## 使用方法

### 启动 CARLA 服务器：
```bash
# 在CARLA安装目录下执行
./CarlaUE4.sh  # Linux/Mac
CarlaUE4.exe   # Windows
```

### 运行自动驾驶系统：
```bash
python main.py --mode auto  # 自动模式（启用感知与决策）
python main.py --mode manual  # 手动模式（Pygame键盘控制）
```

### 数据采集与模型训练：
```bash
# 采集传感器数据（存储至./dataset目录）
python sensor_manager.py --record --output ./dataset

# 训练语义分割模型
python model_trainer.py --data ./dataset --epochs 50
```

## 参数调整指南

| 参数               | 调整范围          | 效果说明                         |
|--------------------|-------------------|----------------------------------|
| `camera_resolution` | 1280x720~1920x1080 | 提高分辨率增强细节（增加计算量） |
| `lidar_points_per_second` | 50000~200000 | 提高值提升点云密度（增加内存占用）|
| `model_batch_size`  | 8~32              | 增大批次加速训练（需更多显存）   |
| `simulation_fps`    | 10~60             | 提高帧率增强实时性（对硬件要求更高）|

## 参考资料
- [CARLA 官方文档](https://carla.readthedocs.io/)
- [Keras 深度学习模型构建指南](https://keras.io/guides/)
- [TensorFlow 模型优化文档](https://www.tensorflow.org/guide/keras/optimizers)
- [语义分割算法综述](https://arxiv.org/abs/1704.06857)