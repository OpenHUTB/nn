# 🚁 AirSim 无人机激光雷达系统：避障与 3D 扫描

> 基于 Microsoft AirSim API 实现的无人机自动避障控制与环境 3D 点云扫描系统。

## 📖 项目简介

本项目利用 Python API 与 Microsoft AirSim 仿真环境交互，通过搭载的 32 线激光雷达（Lidar）获取环境深度数据。主要功能包括：
* **实时障碍物检测**：分析雷达回波数据，判断前方障碍。
* **3D 环境扫描**：获取并在本地保存环境的点云数据。
* **辅助飞行控制**：结合键盘控制与自动避障逻辑，实现安全的仿真飞行。




## 📦 环境依赖 (Dependencies)

在运行代码之前，请确保已安装以下 Python 库：

```bash
pip install airsim numpy keyboard





\## 运行方式

1\. 修改 `settings` 配置文件。
为了确保雷达不被机身遮挡并获得最佳扫描效果，请务必使用以下配置覆盖你的 文档\AirSim\settings.json 文件：
{
  "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/main/docs/settings.md",
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "Vehicles": {
    "Drone_1": {
      "VehicleType": "SimpleFlight",
      "X": 0, "Y": 0, "Z": 0,
      "Sensors": {
        "lidar_1": {
          "SensorType": 6,
          "Enabled": true,
          "Range": 100,
          "NumberOfChannels": 32,
          "PointsPerSecond": 100000,
          "RotationsPerSecond": 10,
          "VerticalFOVUpper": 20,
          "VerticalFOVLower": -45,
          "HorizontalFOVStart": -90,
          "HorizontalFOVEnd": 90,
          "X": 0, "Y": 0, "Z": -1.0,
          "DrawDebugPoints": true,
          "DataFrame": "SensorLocalFrame"
        }
      }
    }
  }
}

2\. 启动虚拟引擎并运行 AirSim。

3\. 运行相关代码。



\## 无人机操控方式



按键	        功能	       说明
W	        前进	       向机头方向水平移动
S	        后退	       向后方水平移动
A	        向左	       向左侧平移 (不改变朝向)
D	        向右	       向右侧平移 (不改变朝向)
Q	        左转   	原地逆时针旋转机头
E	        右转  	原地顺时针旋转机头
↑ (上箭头)	上升  	垂直向上飞行
↓ (下箭头)	下降  	垂直向下飞行
空格  	急刹  	悬停，停止所有移动



