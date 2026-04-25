# 项目简介

本项目是一个基于CARLA仿真器的自动驾驶控制客户端程序，能够连接CARLA服务器并在仿真环境中生成一辆自动驾驶车辆。通过内置的导航代理实现车辆的自动行驶。程序提供多种驾驶行为模式，支持实时显示车辆状态、传感器数据以及多种摄像头视角。

## 主要功能

- **车辆生成**：自动随机选择车辆模型和出生点
- **自动驾驶代理**
  - **RoamingAgent**：随机漫游，无固定目标
  - **BasicAgent**：基本点对点导航
  - **BehaviorAgent**：基于行为规划，支持三种驾驶风格：`cautious`, `normal`, `aggressive`
- **传感器系统**
  - 碰撞检测
  - 车道入侵检测
  - GNSS定位
  - 多种相机：RGB、深度、语义分割
  - 激光雷达
- **实时HUD显示**：显示速度、航向、位置、控制量、碰撞历史、附近车辆等
- **交互控制**：
  - 按 `ESC` 或 `Ctrl+Q` 退出
  - 按 `H` 显示帮助
- **天气切换**：支持多种天气预设
- **循环行驶**：使用 `loop` 参数使车辆到达目标后自动规划新路线

## 系统要求

- **操作系统**：Windows 10/11
- **CARLA版本**：0.9.11（推荐）
- **Python版本**：3.7
- **依赖库**：pygame, numpy, networkx

## 安装与运行

1. **启动CARLA服务器**：
   - 进入CARLA安装目录，双击 `CarlaUE4.exe` 启动服务器。
2. **安装Python依赖**：
   - 执行以下命令安装依赖：
     ```bash
     pip install pygame numpy networkx
     ```
3. **运行客户端**：
   - 打开新终端，进入 `PythonAPI/examples` 目录，执行：
     ```bash
     python main.py
     ```

## 可选参数

- `agent`：选择代理类型 (`Behavior`, `Roaming`, `Basic`)，默认值为 `Behavior`
- `behavior`：行为风格 (`cautious`, `normal`, `aggressive`)，仅 `Behavior` 代理可用，默认值为 `normal`
- `loop`：到达目标后自动循环
- `res`：窗口分辨率（格式：WIDTHxHEIGHT），默认值为 `1280x720`
- `host` 和 `port`：CARLA服务器地址，默认值为 `127.0.0.1:2000`
- `filter`：车辆蓝图过滤器，默认值为 `vehicle`
- `gamma`：相机伽马校正，默认值为 `2.2`
- `seed`：随机种子，用于复现

### 示例

```bash
python main.py agent Behavior behavior aggressive loop