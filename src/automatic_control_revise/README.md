项目简介

本项目是一个基于CARLA仿真器的自动驾驶控制客户端程序 能够连接CARLA服务器并在仿真环境中生成一辆自动驾驶车辆 通过内置的导航代理实现车辆的自动行驶 程序提供多种驾驶行为模式 支持实时显示车辆状态 传感器数据以及多种摄像头视角

主要功能

车辆生成 自动随机选择车辆模型和出生点

自动驾驶代理
RoamingAgent 随机漫游 无固定目标
BasicAgent 基本点对点导航
BehaviorAgent 基于行为规划 支持三种驾驶风格 cautious normal aggressive

传感器系统 碰撞检测 车道入侵检测 GNSS定位 多种相机RGB 深度 语义分割 激光雷达

实时HUD显示 速度 航向 位置 控制量 碰撞历史 附近车辆等

交互控制 按ESC或Ctrl+Q退出 按H显示帮助

天气切换 支持多种天气预设

循环行驶 使用loop参数使车辆到达目标后自动规划新路线

系统要求

操作系统 Windows 10/11

CARLA版本 0.9.11 推荐

Python版本 3.7

依赖库 pygame numpy networkx

安装与运行

启动CARLA服务器 进入CARLA安装目录 双击CarlaUE4.exe 

安装Python依赖 pip install pygame numpy networkx

运行客户端 打开新终端 进入PythonAPI examples目录 执行 python main.py

可选参数

agent Behavior Roaming Basic 选择代理类型 默认Behavior

behavior cautious normal aggressive 行为风格 仅Behavior代理 默认normal

loop 到达目标后自动循环

res WIDTHxHEIGHT 窗口分辨率 默认1280x720

host 和 port CARLA服务器地址 默认127 0 0 1 2000

filter 车辆蓝图过滤器 默认vehicle

gamma 相机伽马校正 默认2 2

seed 随机种子 用于复现

示例

python main.py agent Behavior behavior aggressive loop

文件结构

main.py 主程序
依赖CARLA PythonAPI中的agents navigation模块

注意事项

确保CARLA服务器在运行客户端前已启动

若使用BehaviorAgent 地图需包含有效的道路网络和出生点

首次运行可能需要手动安装carla Python包 参考CARLA官方文档