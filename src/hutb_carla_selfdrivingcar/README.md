# CARLA 自动驾驶模拟系统

基于 CARLA 模拟器的自动驾驶功能演示系统，包含多种智能驾驶辅助功能。

## 功能列表

### 基础功能
- **定速巡航** (`cruise_control.py`) - 保持设定速度行驶
- **碰撞检测** (`collision_detector.py`) - 检测前方障碍物并自动刹车
- **交通灯识别** (`traffic_light_handler.py`) - 检测交通灯状态，红灯停车

### 高级驾驶辅助系统（ADAS）
- **车道保持辅助（LKA）** (`lane_keep_assist.py`) - 通过摄像头检测车道线，自动调整方向盘
- **自适应巡航控制（ACC）** (`adaptive_cruise_control.py`) - 根据前车距离自动调整车速
- **自动紧急刹车（AEB）** (`aeb_system.py`) - 检测到紧急碰撞风险时自动刹车
- **行人保护系统（PPS）** (`pedestrian_protection.py`) - 检测行人并自动刹车
- **自动变道辅助（LCA）** (`lane_change_assist.py`) - 检测相邻车道安全后自动变道
- **自动超车系统** (`auto_overtake.py`) - 检测前方慢车后自动超车
- **交通标志识别（TSR）** (`traffic_sign_recognition.py`) - 识别限速标志并调整车速

### 环境功能
- **天气系统** (`weather_manager.py`) - 支持多种天气类型切换
- **自动泊车** (`auto_parking.py`) - 自动寻找车位并完成泊车

## 快速开始

### 环境要求
- Python 3.8+
- CARLA 模拟器 0.9.14+
- Windows 10/11

### 安装依赖
```bash
pip install carla numpy opencv-python
```

### 运行演示

1. 启动 CARLA 模拟器
2. 运行任意演示程序：

```bash
# 简单驾驶
python simple_drive.py

# 车道保持
python drive_with_lka.py

# 自适应巡航
python drive_with_acc.py

# 天气系统
python drive_with_weather.py

# 自动泊车
python drive_with_parking.py

# 自动紧急刹车
python drive_with_aeb.py

# 行人保护
python drive_with_pedestrian_protection.py

# 自动变道
python drive_with_lane_change.py

# 自动超车
python drive_with_overtake.py

# 交通标志识别
python drive_with_traffic_sign.py

# 完整功能
python main.py
```

## 项目结构

```
hutb_carla_selfdrivingcar/
├── main.py                      # 主程序（完整功能）
├── simple_drive.py              # 简单驾驶版本
├── spawn_car.py                 # 车辆生成模块
├── cruise_control.py            # 定速巡航
├── collision_detector.py       # 碰撞检测
├── traffic_light_handler.py     # 交通灯处理
├── lane_keep_assist.py          # 车道保持辅助（LKA）
├── adaptive_cruise_control.py   # 自适应巡航控制（ACC）
├── aeb_system.py               # 自动紧急刹车（AEB）
├── pedestrian_protection.py     # 行人保护系统（PPS）
├── lane_change_assist.py        # 自动变道辅助（LCA）
├── auto_overtake.py             # 自动超车系统
├── auto_parking.py              # 自动泊车
├── traffic_sign_recognition.py  # 交通标志识别（TSR）
├── weather_manager.py           # 天气系统
└── traffic_manager.py           # 交通场景管理
```

## 功能说明

### 车道保持辅助（LKA）
通过前置摄像头采集道路图像，使用 OpenCV 进行边缘检测和霍夫变换识别车道线，通过 PID 控制器自动调整方向盘，保持车辆在车道中央行驶。

### 自适应巡航控制（ACC）
实时检测前方车辆距离，根据安全距离算法动态调整目标车速，保持与前车的安全跟车距离。

### 自动紧急刹车（AEB）
实时检测前方障碍物距离，根据距离阈值分三级预警（正常→预制动→紧急制动），在危险情况下自动全力刹车。

### 行人保护系统（PPS）
专门针对行人的检测和保护系统，实时检测前方行人距离，在危险距离内自动刹车保护行人安全。

### 自动变道辅助（LCA）
检测左右相邻车道是否有车辆，在安全的情况下自动完成变道操作，提升驾驶便利性。

### 自动超车系统
检测前方是否有慢车阻挡，在满足超车条件时自动向左变道、加速超越、返回原车道，完成完整超车流程。

### 交通标志识别（TSR）
通过前置摄像头实时检测道路上的交通标志，支持限速标志（30/40/50/60/80 km/h）、停车标志、让行标志的识别，并自动调整车速。

### 自动泊车
模拟寻找停车位、靠近车位、倒车入库的完整泊车流程，使用简单的状态机实现自动泊车功能。

### 天气系统
支持晴天、多云、雨天、暴风雨、小雨、夜晚、雨夜等多种天气类型，可自动切换展示不同环境下的驾驶效果。

## 技术栈

- **CARLA 0.9.14** - 自动驾驶模拟器
- **Python 3.8+** - 编程语言
- **OpenCV** - 图像处理和车道线检测
- **NumPy** - 数值计算
- **PID 控制** - 车辆控制算法

## 注意事项

1. 运行前请先启动 CARLA 模拟器
2. 部分功能需要较快的运行帧率，建议在性能较好的电脑上运行
3. 天气系统和交通标志识别功能需要模拟器支持相应元素

## 许可证

MIT License
