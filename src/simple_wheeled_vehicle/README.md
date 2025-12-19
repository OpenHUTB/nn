# 简易轮式小车智能代理

## 项目简介
一个基于PyTorch神经网络与ROS的**简易轮式小车智能代理**，主要面向Carla自动驾驶模拟器。项目实现了从感知到控制的基本自动驾驶流程，模块化设计清晰，便于学习、扩展与二次开发。

## 功能特性
- **环境感知**：处理相机图像，识别车道线、可行驶区域及动态障碍物。
- **决策规划**：基于感知结果进行局部路径规划与动态避障决策。
- **运动控制**：输出转向、油门及刹车指令，控制小车沿规划路径行驶。
- **训练接口**：提供数据采集与模型训练接口，支持强化学习与模仿学习。
- **ROS集成**：通过ROS节点进行模块间通信，支持传感器数据发布与控制指令订阅。

## 环境要求
- **操作系统**：Ubuntu 20.04/22.04（推荐）或 Windows 10/11（WSL2）
- **Python**：3.7 - 3.12
- **核心框架**：PyTorch >= 1.9.0（CPU/GPU版本均可）
- **模拟器**：Carla 0.9.14+
- **可选依赖**：ROS Noetic 或 ROS2 Foxy（用于ROS通信）
- **基础包**：numpy, opencv-python, matplotlib

## 项目结构
```
src/simple_wheeled_agent/
├── main.py                      # 模块主入口 (非ROS启动)
├── main.launch                  # ROS启动配置文件
├── requirements.txt
├── config/
│   └── default.yaml            # 配置文件
├── docs/                       # 模块文档
├── models/                     # 预训练模型存放处
├── scripts/                    # 工具脚本
├── src/                        # 核心源代码
│   ├── perception/
│   │   ├── __init__.py
│   │   ├── image_processor.py # 图像处理与神经网络推理
│   │   └── data_collector.py  # 训练数据采集
│   ├── planning/
│   │   ├── __init__.py
│   │   └── local_planner.py   # 局部路径规划器
│   ├── control/
│   │   ├── __init__.py
│   │   └── pid_controller.py  # 运动控制器
│   └── utils/                  # 通用工具函数
└── tests/                      # 单元测试
```

## 快速开始
1. **安装依赖**:
   ```bash
   cd src/simple_wheeled_agent
   pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
   ```

2. **启动Carla服务端** (在另一个终端):
   ```bash
   ./CarlaUE4.sh -quality-level=Low # 可根据硬件调整画质
   ```

3. **运行代理**:
   ```bash
   python main.py --config config/default.yaml
   ```
   程序将连接Carla，生成车辆，并开始自主驾驶。

4. **(可选) ROS模式**:
   ```bash
   roslaunch simple_wheeled_agent main.launch
   ```

## 主要类和方法
- **`PerceptionAgent` (感知模块)**:
    - `process_image(raw_image)`: 处理原始图像，输出车道与障碍物信息。
    - `collect_training_data()`: 采集带标签的训练数据。

- **`LocalPlanner` (规划模块)**:
    - `compute_path(perception_result, goal)`: 根据感知结果和全局目标计算局部路径。
    - `dynamic_obstacle_avoidance()`: 动态避障算法。

- **`VehicleController` (控制模块)**:
    - `update_control(target_path, current_state)`: 根据目标路径和车辆状态计算控制量。
    - `get_control()`: 返回当前的控制指令（转向、油门、刹车）。

## 配置参数
关键参数可在 `config/default.yaml` 中调整:
```yaml
carla:
  host: localhost
  port: 2000
  timeout: 10.0
vehicle:
  target_speed: 30.0        # 目标速度 (km/h)
  max_steer_angle: 70.0     # 最大转向角 (度)
perception:
  model_path: models/perception_model.pth
  image_width: 640
  image_height: 480
```

## 数据输出
运行过程中会生成以下数据:
1. **控制台输出**：车辆状态、控制指令、故障信息。
2. **日志文件** (位于 `logs/`): 运行日志与性能统计。
3. **数据记录** (训练模式下): 图像与对应控制指令，保存于 `data/train/`。
4. **可视化窗口**：实时显示处理后的图像与规划路径。

## 扩展建议
- **更换模拟器**：参考现有Carla接口，为AirSim或MuJoCo实现对应的环境包装器。
- **改进感知模型**：替换 `perception` 模块中的网络结构（如使用ResNet、Transformer）。
- **尝试新算法**：在 `planning` 模块中实现强化学习（PPO、SAC）或更先进的规划算法。
- **增加传感器**：集成激光雷达（Lidar）或雷达（Radar）数据进行多传感器融合。

## 常见问题
1. **Q: 无法连接到Carla服务器**
   A: 请确认CarlaUE4正在运行，且端口未被占用。检查配置中的 `host` 和 `port`。

2. **Q: 运行时报错 “No module named ‘torch’”**
   A: 请单独安装与你的Python及CUDA版本匹配的PyTorch。

3. **Q: 车辆控制不平稳，抖动严重**
   A: 尝试调整 `config/default.yaml` 中的PID控制参数，或降低 `target_speed`。

4. **Q: 如何用自己的数据训练模型？**
   A: 运行 `python main.py --mode collect` 采集数据，然后使用 `scripts/train_model.py` 进行训练。

5. **Q: ROS节点无法启动**
   A: 确保ROS环境已正确配置 (`source /opt/ros/<distro>/setup.bash`)，且所有ROS依赖已安装。

---
*更详细的API文档请使用 `mkdocs serve` 在本地构建和查看。*