# 小车物体检测仿真项目

## 项目简介

这是一个基于MuJoCo物理引擎的小车物体检测仿真项目。该项目模拟了一辆小车在环境中行驶，并使用LiDAR传感器检测周围物体的功能，同时还具备车内空调温度检测与调节功能。

## 功能特性

1. **物理仿真**：基于MuJoCo的真实物理仿真
2. **小车模型**：包含底盘和四个轮子的简单小车模型
3. **传感器模拟**：LiDAR传感器点云数据生成
4. **物体检测**：检测环境中的障碍物
5. **温度监测**：车内温度实时监测与空调控制系统
6. **数据保存**：保存点云数据和检测结果

## 环境要求

- Python 3.7+
- mujoco
- numpy
- matplotlib

## 项目结构

```
minicarsim/
├── main.py                             # 主程序
├── README.md                           # 说明文档
├── visualize_temperature.py             # 温度数据可视化脚本
├── generate_temperature_charts.py       # 温度图表生成脚本
├── create_temperature_dashboard.py      # 交互式仪表板生成脚本
├── temperature_dashboard.html           # 交互式温度仪表板模板
├── models/
│   └── simple_car.xml                  # 小车和环境模型
└── output/                             # 输出数据目录
    ├── lidar/                          # LiDAR点云数据
    ├── annotations/                    # 物体检测标注
    ├── visualization/                  # 物体识别效果图
    ├── distance_analysis/              # 距离和方位分析图
    └── temperature_analysis/           # 温度分析图表（运行可视化脚本后生成）
```

## 快速开始

1. 安装依赖：
```bash
pip install mujoco numpy matplotlib
```

2. 运行仿真：
```bash
cd src/minicarsim
python main.py
```

3. 观察可视化窗口中的仿真过程

4. 可视化温度数据：
```bash
# 生成静态图表
python generate_temperature_charts.py

# 生成交互式仪表板
python create_temperature_dashboard.py
```

## 代码说明

### 主要类和方法

- `MojocoDataSim`: 主要的仿真类
  - `generate_realistic_lidar_data()`: 生成真实的LiDAR点云数据
  - `detect_objects()`: 检测环境中的物体
  - `get_car_temperature()`: 获取车内温度
  - `adjust_ac_system()`: 调节空调系统
  - `run_simulation()`: 运行仿真主循环

### 配置参数

在`main.py`中可以调整的主要参数：

- `LIDAR_PARAMS`: LiDAR传感器参数
- `AC_PARAMS`: 空调系统参数
- `SIMULATION_FRAMES`: 仿真总帧数

## 模型说明

### simple_car.xml

该文件定义了：
- 地面平面
- 小车模型（底盘和四个轮子）
- 5个彩色障碍物
- 传感器安装位置（LiDAR、摄像头和温度传感器）

## 数据输出

仿真运行后会产生四类数据：

1. **LiDAR点云数据**：保存为`.npy`格式的NumPy数组
2. **物体检测标注**：保存为`.json`格式的标注文件
3. **温度数据**：包含在标注文件中的车内温度和空调功率信息
4. **温度分析图表**：运行可视化脚本后生成的温度变化图表

## 可视化说明

运行仿真后，可以使用多种方式可视化温度数据：

### 静态图表
使用 `generate_temperature_charts.py` 脚本生成：

1. **温度分析图表** (`temperature_analysis_chart.png`)：
   - 显示车内温度和外界温度随时间的变化趋势
   - 显示空调功率调节过程

2. **温度-功率关系图表** (`temperature_power_relation.png`)：
   - 用颜色编码显示温度与空调功率的关系
   - 点的颜色表示空调功率（红色表示高功率，蓝色表示低功率）

### 交互式仪表板
使用 `create_temperature_dashboard.py` 脚本生成：

1. **温度监控仪表板** (`temperature_dashboard.html`)：
   - 交互式温度趋势图
   - 空调功率变化图
   - 温度-功率关系散点图
   - 关键统计数据展示
   - 响应式设计，支持缩放和交互

## 图表解读指南

### 温度分析图表
- **蓝色实线**：车内温度变化曲线
- **红色虚线**：外界温度变化曲线
- **绿色点线**：目标温度线（22°C）
- **绿色填充区域**：空调功率变化情况

### 温度-功率关系图表
- **散点颜色**：表示空调功率（暖色=高功率，冷色=低功率）
- **绿色虚线**：目标温度线（22°C）
- **点的分布**：反映温度控制效果

### 交互式仪表板
- **实时交互**：鼠标悬停查看详细数据
- **缩放功能**：可以放大查看细节
- **响应式设计**：适配不同屏幕尺寸
- **数据统计**：关键指标一目了然

## 扩展建议

1. 添加更多类型的传感器（如摄像头）
2. 实现更复杂的物体检测算法
3. 添加不同的环境地图
4. 实现自主导航功能
5. 添加更多的车辆控制方式
6. 改进空调系统模型，添加湿度检测等功能

## 故障排除

### 常见问题

1. **找不到模型文件**：检查`XML_PATH`是否正确
2. **无法显示可视化窗口**：确保已正确安装MuJoCo
3. **没有检测到物体**：检查小车与物体之间的距离
4. **缺少matplotlib库**：运行`pip install matplotlib`安装

### 支持

如有问题，请提交issue或联系项目维护者。