# 多模态 CARLA 导航避障系统
## 项目简介
本项目基于 CARLA 模拟器与神经网络技术，实现了具备多传感器融合能力的智能车辆导航避障系统。系统集成前视摄像头、第三视角摄像头与激光雷达（LiDAR），通过多模态数据感知环境，结合智能避障算法，实现车辆自主行驶与障碍物规避功能。
## 核心功能
- 多传感器融合感知：集成 RGB 摄像头（前视 + 第三视角）与 32 线激光雷达，全面获取环境信息
- 智能避障算法：基于激光雷达点云数据，分区域检测障碍物，动态选择最优避障方向（左 / 右）
- 可视化监控：实时显示前视画面、第三视角跟随画面、LiDAR 鸟瞰图，叠加关键行驶数据
- 人机交互控制：支持键盘手动干预，兼容自动 / 手动切换
## 环境配置
- 操作系统：Windows 10/11 或 Ubuntu 20.04/22.04
- Python 版本：3.7
- 核心框架：PyTorch
- 模拟器：CARLA3.11
## 依赖安装
- 安装 CARLA 模拟器(参考 CARLA 官网)
- 安装 Python 依赖包：
- ```bash
  pip install -r requirements.txt
  pip install carla numpy opencv-python matplotlib
  pip install setuptools==40.2.0
  pip insatll wheel
  ```
## 快速启动

步骤 1：启动 CARLA 模拟器
- ``` bash
  CarlaUE4.exe -windowed -ResX=800 -ResY=600  #Windows 示例
  ./CarlaUE4.sh -windowed -ResX=800 -ResY=600 #Ubuntu 示例
  ```
步骤 2：运行导航避障系统
- 进入项目根目录
- ```bash
  cd D:\nn
  ```
- 激活虚拟环境
- ``` bash 
  source venv/bin/activate  # Linux/Mac
  venv\Scripts\activate     # Windows
  ```
- 运行主程序
- ``` bash
  python src/robot_navigation_system/main.py
  ```
步骤 3：操作说明
- 按键	功能描述:
- q	退出系统
- w	手动加速（提高油门）
- s	手动减速（降低油门）
- a	手动左转向
- d	手动右转向
- r	重置转向角度（回正）

## 系统架构
1. 环境初始化模块
连接 CARLA 服务器（默认 localhost:2000）。
加载 Town01 地图，设置同步模式（固定时间步长 0.05s）。
配置天气参数（多云、无降水、太阳高度角）。
2. 智能体生成模块
主车辆：特斯拉 Model3（红色），关闭自动驾驶，由自定义算法控制。
障碍物车辆：随机生成 6 辆不同类型车辆，开启自动驾驶，分散分布在地图中。
