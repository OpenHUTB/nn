# 小车物体检测仿真项目

## 项目简介

这是一个基于MuJoCo物理引擎的小车物体检测仿真项目。该项目模拟了一辆小车在环境中行驶，并使用LiDAR传感器检测周围物体的功能。

## 功能特性

1. **物理仿真**：基于MuJoCo的真实物理仿真
2. **小车模型**：包含底盘和四个轮子的简单小车模型
3. **传感器模拟**：LiDAR传感器点云数据生成
4. **物体检测**：检测环境中的障碍物
5. **数据保存**：保存点云数据和检测结果
6. **温度监控**：模拟车内温度变化并可视化
7. **智能空调系统**：根据温度自动开启/关闭空调调节温度
8. **温度调节可视化**：生成多种温度调节相关图表
9. **故障自检系统**：实时监测传感器和执行器状态
10. **故障记录与可视化**：自动生成故障报告和趋势图

## 环境要求

- Python 3.7+
- mujoco
- numpy

## 项目结构

```
minicarsim/
├── main.py              # 主程序
├── README.md            # 说明文档
├── models/
│   └── simple_car.xml   # 小车和环境模型
└── output/              # 输出数据目录
    ├── lidar/           # LiDAR点云数据
    ├── annotations/     # 物体检测标注
    ├── visualization/   # 可视化图表
    │   ├── temperature_trend.png       # 温度变化趋势图
    │   ├── temperature_summary.png     # 温度调节汇总图
    │   ├── ac_control_XXXX.png         # 空调控制状态图
    │   └── thermal_map_XXXX.png        # 环境温度分布图
    └── fault_reports/   # 故障监测报告
        ├── frame_XXXX.json             # 故障详细报告(JSON)
        ├── fault_status_XXXX.png       # 故障状态图
        └── health_trend.png            # 系统健康趋势图
```

## 快速开始

1. 安装依赖：
```bash
pip install mujoco numpy
```

2. 运行仿真：
```bash
cd src/minicarsim
python main.py
```

3. 观察可视化窗口中的仿真过程

## 代码说明

### 主要类和方法

- `MojocoDataSim`: 主要的仿真类
  - `generate_realistic_lidar_data()`: 生成真实的LiDAR点云数据
  - `detect_objects()`: 检测环境中的物体
  - `simulate_temperature_data()`: 模拟车内温度变化
  - `check_and_control_ac()`: 检查并控制空调开关
  - `run_simulation()`: 运行仿真主循环
  - `visualize_ac_control()`: 生成空调控制状态图
  - `generate_temperature_trend()`: 生成温度变化趋势图
  - `generate_temperature_summary_chart()`: 生成温度调节汇总图
  - `generate_fault_report()`: 生成故障报告
  - `visualize_fault_status()`: 生成故障状态图
  - `generate_health_trend()`: 生成健康趋势图

- `FaultMonitor`: 故障监测类
  - `check_system_health()`: 检查系统整体健康状态
  - `_check_sensor_health()`: 检查传感器健康状态
  - `_check_actuator_health()`: 检查执行器健康状态

### 配置参数

在`main.py`中可以调整的主要参数：

- `LIDAR_PARAMS`: LiDAR传感器参数
- `TEMPERATURE_PARAMS`: 温度监控参数
- `FAULT_MONITORING_PARAMS`: 故障监测参数
- `SIMULATION_FRAMES`: 仿真总帧数

## 模型说明

### simple_car.xml

该文件定义了：
- 地面平面
- 小车模型（底盘和四个轮子）
- 5个彩色障碍物
- 传感器安装位置（LiDAR和摄像头）

## 数据输出

仿真运行后会产生四类数据：

1. **LiDAR点云数据**：保存为`.npy`格式的NumPy数组
2. **物体检测标注**：保存为`.json`格式的标注文件
3. **温度调节可视化图表**：
   - 温度变化趋势图 (`temperature_trend.png`)
   - 温度调节汇总图 (`temperature_summary.png`)
   - 空调控制状态图 (`ac_control_XXXX.png`)
   - 环境温度分布图 (`thermal_map_XXXX.png`)
4. **故障监测报告**：
   - 故障详细报告 (`frame_XXXX.json`)
   - 故障状态图 (`fault_status_XXXX.png`)
   - 系统健康趋势图 (`health_trend.png`)

## 扩展建议

1. 添加更多类型的传感器（如摄像头）
2. 实现更复杂的物体检测算法
3. 添加不同的环境地图
4. 实现自主导航功能
5. 添加更多的车辆控制方式
6. 增强空调控制算法（PID控制等）
7. 增加更多类型的故障检测（如通信故障、电源故障等）

## 故障排除

### 常见问题

1. **找不到模型文件**：检查`XML_PATH`是否正确
2. **无法显示可视化窗口**：确保已正确安装MuJoCo
3. **没有检测到物体**：检查小车与物体之间的距离
4. **温度调节效果不明显**：检查空调功率系数和舒适温度范围设置
5. **没有生成故障报告**：检查故障检测阈值设置

### 支持

如有问题，请提交issue或联系项目维护者。