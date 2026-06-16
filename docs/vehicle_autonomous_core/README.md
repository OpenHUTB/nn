# CARLA 自动驾驶辅助功能合集

基于 ****CARLA 自动驾驶仿真平台**** 的多模块车辆辅助驾驶系统，包含车辆生成、手动控制、定速巡航、车道保持、自动紧急制动、激光雷达障碍物检测、多传感器数据采集等功能。整套代码基于 CARLA Python API 开发，可在仿真环境中独立运行、调试与验证，完成自动驾驶****感知、控制****两大基础任务。

模块代码位于仓库的 src/vehicle\_autonomous\_core/

## 目录

-   项目简介
-   安装流程
-   快速开始
-   运行效果
-   核心技术
-   功能模块说明
-   项目结构
-   参考资料

## 项目简介

自动驾驶系统分为感知、决策、控制三大核心环节，车辆运动控制与环境感知是入门开发的基础。本项目以 CARLA 自动驾驶仿真器为运行环境，无需实车即可完成算法验证，实现了一套轻量化、模块化的车辆辅助驾驶程序集。

本项目所有功能均为独立脚本，每一个模块对应一项典型辅助驾驶功能，覆盖从车辆创建、人工操控、自动巡航、车道保持、主动避障到传感器数据采集全流程。

## 安装流程

1.  克隆仓库：

plaintext

git clone https://github.com/OpenHUTB/nn.git  
cd nn/src/hutb\_carla\_selfdrivingcar  

2.  安装 Python 依赖包：

plaintext

pip install carla numpy opencv-python keyboard  

国内网络可加清华镜像：

plaintext

pip install carla numpy opencv-python keyboard -i https://pypi.tuna.tsinghua.edu.cn/simple  

3.  启动 CARLA 仿真器  
    运行 CARLA 主程序，保证本地 [localhost:2000](https://localhost:2000) 端口正常监听，再执行 Python 脚本。

## 快速开始

本项目每个功能均为独立入口文件，统一使用 python 文件名.py 方式运行，执行前务必保证 CARLA 模拟器已启动。

### 车辆批量生成

plaintext

python main\_vehicle\_spawn.py  

### 键盘手动控制车辆

plaintext

python main\_vehicle\_keyboard\_control.py  

### 定速巡航功能

plaintext

python main\_vehicle\_cruise.py  

### 基础车道保持

plaintext

python main\_vehicle\_lane\_keep.py  

### PID 改进版车道保持

plaintext

python main\_vehicle\_lane\_keep\_new.py  

### 自动紧急制动

plaintext

python main\_vehicle\_auto\_brake.py  

### 激光雷达障碍物检测

plaintext

python main\_vehicle\_obstacle\_detector.py  

### 车载相机画面采集

plaintext

python main\_vehicle\_camera.py  

### 相机 + 激光雷达多传感器数据采集

plaintext

python main\_vehicle\_sensor.py  

## 运行效果

### 车辆生成与自动驾驶

程序连接 CARLA 仿真世界，自动在地图出生点生成指定车型车辆，开启自动驾驶后车辆可沿道路正常行驶，支持一次性生成多台不同品牌车辆同时运行。

### 键盘手动操控

通过 W/A/S/D 按键分别实现加速、左转、减速、右转，终端实时打印当前车速、油门、刹车、转向数值，ESC 键退出程序并销毁车辆。

### 定速巡航

车辆起步后自动调速，将车速稳定在设定目标值，对速度偏差进行动态修正，实现匀速行驶效果。

### 车道保持

车辆行驶过程中实时获取地图路点，计算横向偏移量并自动修正转向，使车辆始终保持在车道中央；引入 PID 算法后，转向响应更平稳、偏移量更小。

### 自动紧急制动

模拟前方障碍物场景，车辆检测到危险距离后立即切断油门、施加刹车，完成紧急减速 / 停车。

### 激光雷达障碍物检测

雷达实时输出周围点云数据，识别前方近距离障碍物并在控制台打印告警信息，同时控制车辆主动减速。

### 图像与点云数据采集

相机连续截取行车画面并保存为图片，激光雷达同步采集三维点云，所有数据自动存入指定文件夹，可供后续视觉算法、感知模型训练使用。

## 核心技术

### CARLA 仿真环境交互

所有模块基于 CARLA 官方 Python API 开发，标准调用流程：

1.  建立客户端连接 carla.Client ("[localhost](https://localhost)", 2000)
2.  获取仿真世界、地图、蓝图库
3.  选取车辆 / 传感器蓝图，在出生点生成仿真实体
4.  调用控制接口或传感器回调函数完成业务逻辑
5.  程序结束统一销毁实体，释放资源

### 闭环速度控制（定速巡航）

读取车辆实时速度，与预设目标速度做差值判断，动态调整油门与刹车输出，抑制速度波动，实现匀速行驶。

### 基于路点的车道保持

调用 CARLA 地图接口获取当前位置下一个路点，计算车辆与理想行驶轨迹的横向误差，根据误差大小输出转向角度；进阶版本引入 PID 控制算法，利用比例、微分项优化转向动作，提升行驶平顺性。

### 激光雷达障碍物感知

解析激光雷达原始点云数据，筛选前方有效点云并计算最近距离，根据距离阈值判定危险等级，联动车辆执行减速、刹车动作。

### 多传感器数据读写

利用相机传感器回调截取图像流，结合 OpenCV 完成图片编码与本地保存；激光雷达点云转为数组格式持久化存储，实现多传感器数据同步采集。

## 功能模块说明

表格

| 文件名                               | 核心功能                |
| --------------------------------- | ------------------- |
| main_vehicle_spawn.py             | 批量生成多款车型车辆，统一开启自动驾驶 |
| main_vehicle_control.py           | 车辆基础油门、刹车、转向控制演示    |
| main_vehicle_keyboard_control.py  | 键盘实时操控车辆，附带运行状态打印   |
| main_vehicle_cruise.py            | 实现车辆定速巡航，稳定维持目标车速   |
| main_vehicle_auto_brake.py        | 模拟前方障碍，实现自动紧急制动功能   |
| main_vehicle_lane_keep.py         | 基于地图路点的基础车道保持功能     |
| main_vehicle_lane_keep_new.py     | 搭载 PID 算法的改进版车道保持   |
| main_vehicle_obstacle_detector.py | 激光雷达障碍物检测、告警与主动减速   |
| main_vehicle_camera.py            | 车载 RGB 相机视频帧采集与图片保存 |
| main_vehicle_sensor.py            | 相机 + 激光雷达多传感器同步数据采集 |

## 项目结构

plaintext

src/hutb\_carla\_selfdrivingcar/  
    main\_vehicle\_spawn.py          多车型批量生成模块  
    main\_vehicle\_control.py       车辆基础控制模块  
    main\_vehicle\_keyboard\_control.py 键盘手动控制模块  
    main\_vehicle\_cruise.py        定速巡航模块  
    main\_vehicle\_auto\_brake.py    自动紧急制动模块  
    main\_vehicle\_lane\_keep.py     基础车道保持模块  
    main\_vehicle\_lane\_keep\_new.py PID 优化车道保持模块  
    main\_vehicle\_obstacle\_detector.py 激光雷达障碍物检测模块  
    main\_vehicle\_camera.py        车载相机图像采集模块  
    main\_vehicle\_sensor.py        多传感器融合采集模块  
    README.md                     项目整体说明文档  

## 参考资料

1.  CARLA 自动驾驶仿真平台官方文档：[https://carla.org/](https://carla.org/)
2.  CARLA Python API 开发手册
3.  自动驾驶车辆运动控制与 PID 控制理论
4.  激光雷达、车载相机传感器应用开发资料

### 使用说明

1.  全选上面所有内容，直接复制；
2.  粘贴到 ****Word、WPS、飞书文档、语雀、微信公众号、网页富文本编辑器**** 均可自动识别格式；
3.  代码块区域：在富文本编辑器中可手动标记为「代码样式 / 等宽字体」，表格会自动保留行列结构；
4.  如需调整字体、字号、加粗、配色，在对应编辑器里可视化编辑即可。