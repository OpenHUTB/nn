---
AIGC:
    ContentProducer: Minimax Agent AI
    ContentPropagator: Minimax Agent AI
    Label: AIGC
    ProduceID: e052170842fbffc7d21baedf4eb60a91
    PropagateID: e052170842fbffc7d21baedf4eb60a91
    ReservedCode1: 30450220634cdb7d01a90f09d27c8175ddbead6c2e00eb6885f4bf068b71a5e68462deca022100bae83365aff19667c6a3d81e38e6b421807893b6f6908a22e839588a0946a346
    ReservedCode2: 3046022100a7bfdf8c0b714f1dba19461db6b74519f502d4673a07347747fb0cf56e840d95022100854fe1d24afa96ab2ea3fae7b6bc2b8d75f2021846f4dce4f4743a9680d68e0d
---

# 智能无人车导航系统 

![无人车导航系统](imgs/chinese_nav_0.png)

> 基于ROS的智能无人车自主导航系统，支持路径规划、障碍物避让、实时定位等功能

## 项目简介

本项目是一个基于ROS (Robot Operating System) 框架开发的智能无人车导航系统，集成了先进的SLAM算法、路径规划、动态避障等功能模块，实现了完全自主的室内外导航能力。

### 核心特性

- 实时建图 (SLAM): 基于激光雷达的实时环境建图
- 智能路径规划: 支持全局路径规划和局部路径优化
- 动态避障: 实时检测障碍物并调整路径
- 精确定位: 多传感器融合定位系统
- 多种控制模式: 支持手动控制、自动导航、目标点导航
- 实时监控: 可视化界面实时显示系统状态

##  快速开始

### 系统要求

- **操作系统**: Ubuntu 18.04/20.04 LTS
- **ROS版本**: Melodic/Noetic
- **Python**: 2.7/3.x
- **硬件要求**:
  - 激光雷达 (16线/32线)
  - 深度相机
  - IMU传感器
  - 编码器

### 安装步骤

```bash
# 1. 创建工作空间
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src

# 2. 克隆项目
git clone https://github.com/your-repo/autonomous-vehicle.git

# 3. 安装依赖
cd autonomous-vehicle
rosdep install --from-paths src --ignore-src -r -y

# 4. 编译项目
cd ~/catkin_ws
catkin_make

# 5. 加载环境变量
source devel/setup.bash
```

### 运行演示

```bash
# 启动仿真环境
roslaunch autonomous_vehicle simulation.launch

# 启动真实机器人
roslaunch autonomous_vehicle robot.launch

# 启动导航系统
roslaunch autonomous_vehicle navigation.launch
```

## 运行效果图

### 1. ROS导航仿真界面

(imgs/ros_navigation_4.png)
*图1: ROS RViz环境下的导航仿真界面，实时显示地图、路径规划和机器人位置*

### 2. 路径规划可视化

![路径规划](imgs/path_planning_6.png)
*图2: 智能路径规划系统，绿色线条显示全局规划路径，红色线条显示局部优化路径*

### 3. 障碍物避让演示

![障碍物避让](imgs/obstacle_avoidance_8.jpg)
*图3: 实时障碍物检测与避让，黄色区域为检测到的障碍物，蓝色区域为安全路径*

### 4. 中文导航界面

![中文导航](imgs/chinese_nav_8.png)
*图4: 中文导航系统界面，支持目标点设定、路径显示和状态监控*

### 5. 实时地图构建

![地图构建](imgs/chinese_nav_7.png)
*图5: 实时SLAM地图构建过程，显示环境特征点和机器人轨迹*

##  系统架构

```
autonomous_vehicle/
├── src/
│   ├── perception/          # 感知模块
│   │   ├── obstacle_detection/
│   │   ├── lane_detection/
│   │   └── traffic_light_detection/
│   ├── localization/        # 定位模块
│   │   ├── slam/
│   │   └── ekf_localization/
│   ├── planning/           # 规划模块
│   │   ├── global_planner/
│   │   └── local_planner/
│   ├── control/            # 控制模块
│   │   ├── pid_controller/
│   │   └── pure_pursuit/
│   └── utils/              # 工具模块
├── launch/                 # 启动文件
├── config/                 # 配置文件
├── urdf/                   # 机器人模型
├── rviz/                   # 可视化配置
└── scripts/                # 脚本文件
```

## 性能指标

| 指标项 | 性能参数 |
|--------|----------|
| 最大速度 | 2.0 m/s |
| 最小转弯半径 | 0.5 m |
| 定位精度 | ±5 cm |
| 避障距离 | 1.0 m |
| 地图分辨率 | 5 cm/pixel |
| 路径规划频率 | 10 Hz |

 使用场景

- 室内服务机器人: 办公楼、医院、酒店等场所的自主导航
- 工业AGV**: 工厂内的物料搬运和自动化运输
-  室外巡检机器人**: 园区巡检、安防监控等应用
-  教育研究**: 机器人学习和算法验证平台

##  API文档

### 主要话题

| 话题名称 | 消息类型 | 功能描述 |
|----------|----------|----------|
| `/cmd_vel` | `geometry_msgs/Twist` | 机器人运动控制命令 |
| `/scan` | `sensor_msgs/LaserScan` | 激光雷达数据 |
| `/amcl_pose` | `geometry_msgs/PoseWithCovarianceStamped` | 机器人位姿信息 |
| `/move_base/goal` | `geometry_msgs/PoseStamped` | 导航目标点 |

### 主要服务

| 服务名称 | 服务类型 | 功能描述 |
|----------|----------|----------|
| `/slam_gmapping/map` | `nav_msgs/GetMap` | 获取地图数据 |
| `/static_map` | `nav_msgs/GetMap` | 获取静态地图 |

## 配置说明

### 导航参数配置

```yaml
# costmap_common_params.yaml
obstacle_radius: 0.2
inflation_radius: 0.5
max_vel_x: 0.5
min_vel_x: -0.5
max_vel_theta: 1.0
```

### 激光雷达配置

```yaml
# lidar_params.yaml
min_range: 0.2
max_range: 30.0
scan_frequency: 5.0
angular_resolution: 0.25
```

##  故障排除

### 常见问题

1. **激光雷达数据丢失**
   - 检查激光雷达连接
   - 确认串口权限设置

2. **定位不准确**
   - 校准IMU传感器
   - 检查编码器数据

3. **路径规划失败**
   - 检查地图数据完整性
   - 确认目标点可达性

### 日志查看

```bash
# 查看系统日志
roslaunch autonomous_vehicle log.launch

# 实时监控话题数据
rostopic echo /cmd_vel
```
### 开发环境搭建

```bash
# 安装开发工具
sudo apt install python3-catkin-tools python3-osrf-pycommon

# 克隆开发分支
git checkout -b feature/your-feature
```