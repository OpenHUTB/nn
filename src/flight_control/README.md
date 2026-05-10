# 无人机飞行控制系统

## 项目介绍

这是一个基于 AirSim 模拟器的无人机飞行控制系统，支持命令行控制、GUI 控制和地图显示功能。该系统可以实时显示无人机的位置和飞行轨迹，提供直观的飞行控制界面。同时集成了 YOLO 目标检测和强化学习功能，支持无人机自主导航和目标识别。

## 功能特性

- **多模式控制**：支持命令行控制和 GUI 控制
- **地图显示**：实时显示无人机位置和飞行轨迹
- **实时状态**：显示无人机的位置、速度、高度等状态信息
- **传感器数据**：显示气压等传感器数据
- **手动控制**：支持键盘控制无人机飞行
- **自动功能**：支持起飞、降落、悬停、返航等自动操作
- **参数设置**：可调整飞行速度和高度
- **日志记录**：记录系统运行日志
- **YOLO 目标检测**：集成 YOLOv8 进行实时目标检测，支持可视化显示
- **强化学习**：支持 PPO、DQN、A2C 等强化学习算法训练和评估

## 系统要求

- **操作系统**：Windows 10/11
- **Python**：Python 3.7 或更高版本
- **AirSim**：Microsoft AirSim 模拟器
- **依赖库**：
  - airsim
  - pynput
  - tkinter (Python 标准库)
  - ultralytics (YOLO 目标检测)
  - opencv-python (图像处理)
  - stable-baselines3 (强化学习)
  - torch (深度学习框架)
  - gym (OpenAI 强化学习环境)

## 安装说明

1. **安装 Python**：从 [Python 官网](https://www.python.org/downloads/) 下载并安装 Python 3.7+。
2. **安装依赖库**：
   ```bash
   pip install airsim pynput ultralytics opencv-python stable-baselines3 torch gym
   ```
   或使用项目中的 requirements.txt：
   ```bash
   pip install -r requirements.txt
   ```
3. **安装 AirSim**：从 [AirSim 官网](https://microsoft.github.io/AirSim/) 下载并安装 AirSim 模拟器。
4. **下载项目代码**：将项目代码下载到本地。

## 使用方法

### 启动 AirSim 模拟器

1. 启动 AirSim 模拟器。
2. 选择 "Neighborhood" 环境（AirSimNH）。
3. 等待模拟器完全加载。

### 运行飞行控制系统

#### 方法 1：使用批处理脚本

双击运行 `run_flight.bat` 文件，该脚本会自动启动带有 GUI 和地图显示的飞行控制系统。

#### 方法 2：使用命令行

根据需要选择以下命令之一：

- **仅使用命令行控制**：
  ```bash
  python src\flight_control\main.py
  ```
- **启用 GUI 控制**：
  ```bash
  python src\flight_control\main.py --gui
  ```
- **启用地图显示**：
  ```bash
  python src\flight_control\main.py --map
  ```
- **同时启用 GUI 和地图显示**：
  ```bash
  python src\flight_control\main.py --gui --map
  ```

## 控制指令

- **W**：前进
- **S**：后退
- **A**：向左
- **D**：向右
- **Z**：上升
- **X**：下降
- **H**：悬停
- **B**：返航
- **ESC**：退出并降落

## 项目结构

```
flight_control/
├── config.json          # 配置文件
├── gui.py              # GUI 界面
├── main.py             # 主控制程序
├── map_display.py      # 地图显示模块
├── reinforcement_learning/  # 强化学习相关模块
│   ├── drone_env.py    # 无人机 Gym 环境
│   ├── ppo_model.py    # PPO 模型定义和训练
│   ├── train.py        # 训练脚本
│   ├── run_model.py    # 模型运行脚本
│   ├── evaluate.py     # 模型评估脚本
│   ├── yolo_inference.py  # YOLO 推理模块
│   ├── run_yolo_demo.py    # YOLO 演示脚本
│   └── run_yolo_demo.bat   # YOLO 演示批处理脚本
├── run_flight.bat      # 启动脚本
├── requirements.txt    # 依赖文件
└── README.md           # 项目说明
```

## 模块说明

- **main.py**：主控制程序，负责连接 AirSim 模拟器、处理用户输入、控制无人机飞行。
- **gui.py**：GUI 界面，显示无人机状态和传感器数据，提供控制按钮。
- **map_display.py**：地图显示模块，实时显示无人机位置和飞行轨迹。
- **run_flight.bat**：启动脚本，自动启动带有 GUI 和地图显示的飞行控制系统。
- **config.json**：配置文件，存储系统配置信息，包括强化学习和 YOLO 相关配置。
- **reinforcement_learning/drone_env.py**：无人机 Gym 环境，封装了 AirSim 模拟器的交互接口，支持强化学习训练。
- **reinforcement_learning/ppo_model.py**：PPO 模型定义，包含模型创建、训练、保存和加载功能。
- **reinforcement_learning/train.py**：强化学习训练脚本，用于训练无人机自主飞行策略。
- **reinforcement_learning/run_model.py**：模型运行脚本，用于加载训练好的模型并控制无人机飞行。
- **reinforcement_learning/evaluate.py**：模型评估脚本，用于评估训练好的模型性能。
- **reinforcement_learning/yolo_inference.py**：YOLO 推理模块，封装了 YOLOv8 模型的加载和推理功能。
- **reinforcement_learning/run_yolo_demo.py**：YOLO 目标检测演示脚本，提供图形界面显示检测结果。
- **reinforcement_learning/run_yolo_demo.bat**：YOLO 演示批处理脚本，自动检查环境并启动演示。

## 飞行模式

### 手动模式

使用键盘控制无人机飞行：

- 使用 WASD 键控制无人机的前后左右移动
- 使用 ZX 键控制无人机的升降
- 使用 H 键让无人机悬停
- 使用 B 键让无人机返航到原点

### 自动模式

通过 GUI 界面的按钮执行自动操作：

- **起飞**：无人机自动起飞并到达指定高度
- **降落**：无人机自动降落
- **悬停**：无人机保持当前位置悬停
- **返航**：无人机返回起飞点

## 地图显示

地图显示模块会：

- 显示无人机的实时位置
- 记录并绘制无人机的飞行轨迹
- 提供网格背景和坐标轴参考
- 支持在没有 AirSim 模拟器的情况下显示模拟位置

## YOLO 目标检测

### 功能说明

- **实时目标检测**：使用 YOLOv8 模型实时检测环境中的目标
- **可视化显示**：在图像上显示边界框、类别和置信度
- **性能指标**：显示 FPS 信息，评估检测性能
- **模型选择**：支持默认 YOLOv8n 模型和自定义模型

### 配置选项

在 `config.json` 文件中可以配置 YOLO 相关参数：

```json
{
  "reinforcement_learning": {
    "use_yolo": true,
    "yolo_model_path": null,
    "yolo_conf_threshold": 0.5,
    "yolo_iou_threshold": 0.45
  }
}
```

## 强化学习

### 功能说明

- **多种算法支持**：支持 PPO、DQN、A2C 等主流强化学习算法
- **图像输入**：支持直接使用摄像头图像作为观察输入
- **YOLO 集成**：可选集成 YOLO 目标检测，增强环境感知能力
- **目标穿越任务**：支持无人机自主穿越目标框的强化学习训练
- **模型评估**：提供模型评估脚本，评估训练效果

### 使用方法

#### 训练模型

```bash
cd src\flight_control
python reinforcement_learning\train.py
```

#### 运行训练好的模型

```bash
python reinforcement_learning\run_model.py --model models/ppo_drone.zip
```

#### 评估模型性能

```bash
python reinforcement_learning\evaluate.py --model models/ppo_drone.zip
```

### 配置选项

在 `config.json` 文件中可以配置强化学习相关参数：

```json
{
  "reinforcement_learning": {
    "enabled": true,
    "model": "ppo",
    "model_path": null,
    "use_yolo": false,
    "yolo_model_path": null,
    "yolo_conf_threshold": 0.5,
    "yolo_iou_threshold": 0.45
  }
}
```

## 故障排除

### 连接失败

- 确保 AirSim 模拟器已启动
- 确保 AirSim 模拟器处于正常运行状态
- 检查网络连接是否正常

### 地图显示不更新

- 确保启用了 `--map` 参数
- 检查无人机是否正在移动
- 确保 AirSim 模拟器与控制程序之间的通信正常

### GUI 启动失败

- 确保安装了所有依赖库
- 检查 Python 版本是否符合要求
- 确保系统支持 tkinter 图形界面

### YOLO 检测失败

- 确保安装了 `ultralytics` 库
- 首次运行时会自动下载 YOLOv8 模型（约 6MB）
- 确保 AirSim 模拟器正在运行并提供图像数据

### 强化学习训练失败

- 确保安装了 `stable-baselines3` 和 `torch` 库
- 检查 AirSim 模拟器是否正常运行
- 确保有足够的磁盘空间保存模型文件

  <br />

***

**享受飞行控制的乐趣！** 🚁
