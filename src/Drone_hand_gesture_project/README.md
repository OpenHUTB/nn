# 手势控制无人机项目

基于计算机视觉和 AirSim 模拟器的手势控制无人机系统。支持实时手势识别、自动飞行、数据采集等功能。

## 功能特性

- ✅ 实时手势识别（MediaPipe）
- ✅ AirSim 模拟器集成
- ✅ 3D 可视化
- ✅ 飞行数据记录与分析
- ✅ 传感器数据采集
- ✅ 自动飞行模式（圆形、8 字形、方形）
- ✅ 键盘控制备用

## 项目结构

```text
Drone_hand_gesture_project/
├── main.py                    # 主程序入口
├── airsim_controller.py       # AirSim 控制器
├── drone_controller.py        # 无人机控制器
├── flight_recorder.py         # 飞行数据记录器
├── sensor_data_collector.py   # 传感器数据采集
├── auto_flight.py             # 自动飞行模式
├── test_airsim.py             # AirSim 连接测试
├── simulation_3d.py          # 3D 可视化
├── physics_engine.py         # 物理仿真引擎
├── gesture_detector.py       # 基础手势检测器
├── gesture_detector_enhanced.py  # 增强手势检测器
├── gesture_classifier.py     # 手势识别分类器
├── gesture_data_collector.py # 手势图像数据收集
├── train_gesture_model.py    # 训练识别模型
├── start_airsim.bat          # Windows 启动脚本
├── AIRSIM_STARTUP.md         # AirSim 启动指南
├── dataset/                  # 数据集目录
│   ├── raw/                  # 原始数据
│   ├── processed/            # 处理后的数据
│   └── models/               # 训练好的模型
└── requirements.txt          # 依赖列表
```

## 快速开始

### Windows 用户

1. **启动 AirSim 模拟器**
   ```bash
   # 双击运行
   d:\机械学习\air\Blocks\WindowsNoEditor\Blocks.exe
   ```

2. **运行手势控制程序**
   ```bash
   cd src/Drone_hand_gesture_project
   start_airsim.bat
   ```

### Linux/Mac 用户

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **启动 AirSim**（需要 Wine 或原生支持）

3. **运行程序**
   ```bash
   python main.py
   ```

## 使用方法

### 手势控制

| 手势 | 动作 |
|------|------|
| 张开手掌 | 悬停 |
| 食指向上 | 上升 |
| 食指向下 | 下降 |
| 指向左右 | 左右移动 |
| 握拳 | 停止 |

### 键盘控制

- `W/S` - 上升/下降
- `A/D` - 左移/右移
- `F/B` - 前进/后退
- `空格` - 起飞/降落
- `C` - 连接无人机
- `ESC` - 退出

### 自动飞行

```python
from auto_flight import AutoFlight

auto = AutoFlight()
auto.connect()
auto.takeoff()

# 圆形飞行
auto.fly_circle((0, 0), radius=3.0, altitude=2.0)

# 8 字形飞行
auto.fly_figure8((0, 0), width=4.0, height=2.0, altitude=2.0)

auto.land()
```

### 数据采集

```python
from sensor_data_collector import SensorDataCollector

collector = SensorDataCollector()
collector.connect()

# 采集 IMU 数据
imu_data = collector.collect_imu_data()

# 连续采集
data = collector.continuous_collect(duration=10.0)

# 保存数据
collector.save_all_data()
```

## 飞行数据记录

```python
from flight_recorder import FlightRecorder

recorder = FlightRecorder()
recorder.start_recording()

# ... 飞行操作 ...

recorder.stop_recording()
recorder.save_to_csv()  # 或 save_to_npy(), save_to_json()
recorder.plot_trajectory()  # 绘制 3D 轨迹
```

## 测试

```bash
# 测试 AirSim 连接
python test_airsim.py

# 测试飞行记录器
python flight_recorder.py

# 测试传感器采集
python sensor_data_collector.py
```

## 环境要求

- Python 3.7-3.12
- OpenCV 4.5+
- MediaPipe 0.10+
- AirSim 1.7+（可选）
- NumPy, Pandas, Matplotlib

安装依赖：
```bash
pip install -r requirements.txt
```

## 贡献

本项目遵循 nn 仓库贡献指南：

1. 模块位于 `src/模块名` 目录
2. 入口文件为 `main.py`
3. 提供完整的 README 说明
4. 代码符合 PEP 8 规范
5. 提供测试和文档

## 参考项目

本项目基于以下开源项目开发：

本项目基于以下开源项目开发：

- [Autonomous Drone Hand Gesture Project](https://github.com/chwee/AutonomusDroneHandGestureProject)
  - 原始手势控制无人机项目
  - 提供了基础架构和实现思路

- [MediaPipe Hands](https://github.com/google/mediapipe)
  - Google开源的手部关键点检测框架
  - 本项目使用其进行实时手势识别