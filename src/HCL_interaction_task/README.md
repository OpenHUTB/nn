# 生物模型交互与ROS 2集成项目

## 项目概述

本项目融合了基于MuJoCo的人体上肢物理仿真与ROS 2机器人操作系统，实现了兼具高精度手势交互与分布式通信能力的综合仿真系统。项目核心包含两大功能模块：
1. **食指指向追踪**（`mobl_arms_index_pointing`）：通过MediaPipe手部追踪驱动食指指向指定目标（如小球）的仿真功能
- **功能演示**：
![ezgif-72b271013158aed5](https://github.com/user-attachments/assets/937c5994-4cfe-42ae-8f77-2eb0ebb1b41a)

2. **面板选择反应**（`mobl_arms_index_choice_reaction`）：新增的多目标面板交互模块，支持食指对虚拟面板进行选择与反应测试
- **功能演示**：
![ezgif-283f8d513dd75e78](https://github.com/user-attachments/assets/6edd413e-8566-4b36-a8c0-fd6650ceee10)


## 主要特点

### 通用特点
- **高精度骨骼模型**：包含上肢完整骨骼结构，细化手部掌骨、指骨（近节/中节/远节）的几何与物理参数
- **增强型肌肉驱动**：基于FDPI、EDM等食指相关肌肉群的肌腱路径定义，提升动作真实性
- **实时手势映射**：集成MediaPipe手部追踪，将真实手势实时映射到虚拟模型
- **ROS 2分布式通信**：通过ROS 2节点实现关节数据发布、感知数据处理与外部系统交互
- **模块化设计**：保持与[User-in-the-Box](https://github.com/User-in-the-Box/user-in-the-box)原项目的兼容性

### 食指指向模块（`mobl_arms_index_pointing`）
- **目标追踪功能**：支持配置单一目标坐标（如小球），实现食指自动指向指定位置
- **动态轨迹规划**：根据目标位置实时计算手指运动轨迹，确保平滑自然的指向动作

### 面板选择反应模块（`mobl_arms_index_choice_reaction`）
- **多目标交互面板**：虚拟面板上可配置多个选择目标，支持自定义布局与样式
- **反应时间记录**：精确记录从面板目标出现到食指选择的反应时长，用于交互性能分析
- **选择有效性判定**：自动检测食指与目标区域的接触状态，判断选择有效性
- **动态目标生成**：支持随机/序列生成面板目标，模拟复杂选择任务
- **任务逻辑闭环**：通过task.py实现奖励机制、超时判断、任务重置的完整逻辑闭环
- **友好状态提示**：终端实时输出任务进度、奖励数值、碰撞状态，便于调试与分析



## 目录结构

```
HCL_interaction_task/
├── mobl_arms_index_pointing/       # 食指指向追踪模块
│   ├── config.yaml                 # 指向仿真参数配置
│   ├── simulation.xml              # 指向功能模型定义
│   ├── evaluator.py                # 指向模块程序入口
│   ├── simulator.py                # 指向仿真核心逻辑
│   ├── assets/                     # 指向模块模型资源（网格/纹理）
│   └── mujoco_ros_demo/            # 指向模块ROS 2功能包
├── mobl_arms_index_choice_reaction/ # 面板选择反应模块
│   ├── config.yaml                 # 面板交互参数配置
│   ├── simulation.xml              # 面板功能模型定义
│   ├── main.py                     # 面板模块程序入口
│   ├── simulator.py                # 面板仿真核心逻辑
│   ├── task.py                     # 面板任务逻辑核心
│   └── assets/                     # 面板模块模型资源（含面板纹理）
├── README.md                       # 项目说明文档
└── requirements.txt                # 项目依赖清单
```


## 模型结构

### 食指指向模型（`mobl_arms_index_pointing/simulation.xml`）
- 核心组件：上肢骨骼结构、肌肉肌腱系统、目标标记点（小球）
- 关键定义：手指关节活动范围、肌肉附着点、目标位置约束

### 面板选择模型（`mobl_arms_index_choice_reaction/simulation.xml`）
- 核心组件：继承基础上肢模型，新增交互面板几何体、多目标区域标记
- 关键定义：面板碰撞属性、目标区域触发条件、手指接触检测参数



## 核心模块说明

### 面板任务逻辑核心（`mobl_arms_index_choice_reaction/task.py`）
#### 1. 模块作用
作为面板选择反应模块的「大脑」，负责串联仿真流程与交互逻辑，实现从任务初始化到结果判定的完整闭环，无需侵入修改仿真核心代码，保持模块解耦性。

#### 2. 核心功能
| 功能项                | 详细描述                                                                 |
|-----------------------|--------------------------------------------------------------------------|
| 任务初始化/重置       | 支持任务参数加载与状态重置，实现多轮任务自动循环运行                     |
| 指尖-目标碰撞检测     | 精准识别食指远节（`hand_2distph`）与面板目标（`button-x`）的接触状态     |
| 奖励机制设计          | 正确选择目标奖励（默认10.0）、超时失败扣分（默认-5.0）、动作成本惩罚（可选） |
| 超时判断              | 基于配置的超时步数/时间，自动判定任务失败并触发重置                       |
| 实时状态反馈          | 每50步输出任务进度、剩余步数、当前奖励，便于实时监控仿真过程             |
| 兼容无执行器模型      | 对无肌肉/执行器驱动的模型提供友好提示，不影响任务核心逻辑运行             |

#### 3. 关键类与方法
- **`ChoicePanelTask` 类**：任务逻辑封装核心，关联仿真器实例与配置参数
  - `__init__`：初始化任务参数（目标按钮、奖励值、超时步数）
  - `reset()`：重置任务状态，用于多轮任务循环
  - `update()`：每步仿真调用，更新任务状态（碰撞检测、奖励计算、超时判断）
  - `_check_button_contact()`：内部核心方法，实现指尖与目标的碰撞检测

#### 4. 配置依赖
任务参数完全读取自`config.yaml`，支持灵活配置，无需修改代码：
```yaml
# 任务相关配置（已整合到config.yaml）
target_button: 0        # 目标按钮ID（0-3对应红/绿/蓝/黄）
button_reward: 10.0     # 正确选择奖励值
effort_cost: 0.01       # 动作成本惩罚系数（有执行器时生效）
timeout: 600            # 任务超时步数（对应仿真时长）
```


## 系统要求

- Ubuntu 22.04 LTS (ROS 2 Humble)
- Python 3.8+（推荐3.10.18）
- MuJoCo 2.3.0+（测试版本3.3.7）
- 依赖库：OpenCV、MediaPipe、NumPy、PyYAML
- 环境管理：Conda 环境管理器
- 额外依赖（C++）：libglfw3-dev、libyaml-cpp-dev、libeigen3-dev


## 安装步骤

### 1. 克隆仓库（如适用）

```bash
git clone https://github.com/lbxlb/nn.git
cd HCL_interaction_task
```

### 2. 环境配置

```bash
# 创建并激活conda环境
conda create -n mjoco_ros python=3.10
conda activate mjoco_ros

# 安装依赖
pip install -r requirements.txt
# 或手动安装
pip install mujoco mediapipe numpy opencv-python pyyaml

# 安装ROS 2相关依赖（如使用ROS功能）
sudo apt install ros-humble-ros-base
```

### 3. 模型资源准备

1. 下载完整模型文件集（含精细STL模型和纹理）：
   [完整模型文件集网盘链接](链接: https://pan.baidu.com/s/1sA0BgEPRgxXTqe6ZdEm7Sg?pwd=rq8e 提取码: rq8e)

2. 解压资源文件到对应模块的assets目录：
   - 指向模块：`mobl_arms_index_pointing/assets/`
   - 面板模块：`mobl_arms_index_choice_reaction/assets/`


## 配置说明

### 食指指向模块配置（`mobl_arms_index_pointing/config.yaml`）

```yaml
dt: 0.05                # 仿真步长
render_mode: "human"    # 渲染模式（"human"/"offscreen"）
resolution: [1280, 960] # 窗口分辨率
target_pos: [0.4, 0, 0.7] # 目标小球坐标
```

### 面板选择模块配置（`mobl_arms_index_choice_reaction/config.yaml`）

```yaml
dt: 0.05                # 仿真步长
render_mode: "human"    # 渲染模式
resolution: [1280, 960] # 窗口分辨率
panel_pos: [0.5, 0, 0.6] # 面板位置
target_count: 3         # 面板目标数量
target_size: 0.05       # 目标区域大小
reaction_timeout: 5.0   # 反应超时时间（秒）
# 任务逻辑配置（新增）
target_button: 0        # 目标按钮ID（0-3）
button_reward: 10.0     # 正确选择奖励值
effort_cost: 0.01       # 动作成本惩罚系数
timeout: 600            # 任务超时步数
```


### ROS 2功能包结构

```
mujoco_ros_demo/
├── config/
│   ├── assets/           # 3D模型文件(STL格式)
│   ├── humanoid.xml      # 机器人模型配置文件
│   └── config.yaml       # 仿真参数配置文件（C++版本）
├── launch/
│   └── main.launch.py    # ROS2启动文件（支持Python/C++节点）
├── mujoco_ros_demo/      # Python节点目录
│   ├── __init__.py
│   ├── mujoco_publisher.py   # Python版：发布关节角度数据的节点
│   └── data_subscriber.py    # Python版：订阅并处理数据的节点
├── mujoco_demo_cpp/      # C++节点目录（新增）
│   ├── simulator.cpp/.hpp    # C++版：MuJoCo仿真核心逻辑
│   ├── mujoco_publisher.cpp  # C++版：发布关节角度数据的节点
│   ├── data_subscriber.cpp   # C++版：订阅并处理数据的节点
│   ├── data_acquire.cpp      # C++版：感知数据采集节点
│   └── perception_node.cpp   # C++版：感知数据处理节点
├── CMakeLists.txt        # C++编译配置文件（新增）
├── package.xml           # ROS2包配置文件（更新依赖）
├── setup.py              # Python包安装配置
```

## 节点说明

1. **MujocoPublisher** (mujoco_publisher.py/mujoco_publisher.cpp)
   - 功能：加载MuJoCo模型并发布关节角度数据
   - 发布主题：/joint_angles (std_msgs/Float64MultiArray)
   - 参数：model_path - 机器人模型文件路径

2. **DataSubscriber** (data_subscriber.py/data_subscriber.cpp)
   - 功能：订阅关节角度数据并计算平均值
   - 订阅主题：/joint_angles (std_msgs/Float64MultiArray)
   - 输出：终端打印关节角度及平均值

3. **DataAcquire** (data_acquire.py)
   - 功能：采集外部传感器或设备数据（如手势追踪原始数据）
   - 发布主题：/raw_sensor_data (自定义消息类型)
   - 支持：MediaPipe原始数据、外部传感器输入等
![ezgif-59ef5d64b961194d](https://github.com/user-attachments/assets/8baece71-eefd-40f8-8197-a92a7c6f9d02)
4. **PerceptionNode** (perception_node.py)
   - 功能：处理感知数据，实现手势识别与解析
   - 订阅主题：/raw_sensor_data
   - 发布主题：/processed_gesture (包含解析后的手势指令)
![ezgif-40d1cc7854d18dd9](https://github.com/user-attachments/assets/3510e8f7-bf61-466b-9bae-2a55a86406e0)

6. **Main** (main.launch.py)
  - 功能：启动ROS 2系统并连接各个节点
  - 启动节点：MujocoPublisher、DataSubscriber
![ezgif-7dec2645d82e4788](https://github.com/user-attachments/assets/97475b38-4876-409b-b52d-ce0e535c520b)



## 使用方法

### 1. 基础仿真运行

```bash
python evaluator.py --config config.yaml --model simulation.xml
```

### 2. ROS 2系统运行

```bash
# 构建项目
colcon build
source install/setup.bash

# 启动ROS 2节点(python)
ros2 launch mujoco_ros_demo main.launch.py

# 运行数据采集节点(C++)
ros2 run mujoco_ros_demo data_acquire_cpp
```

### 3. 查看运行状态

- 查看节点信息：`ros2 node list`
- 查看话题信息：`ros2 topic list`
- 可视化工具：`rqt`

## 项目来源/参考

- [MuJoCo](https://github.com/deepmind/mujoco) - 高性能物理引擎
- [ROS 2](https://github.com/ros2) - 机器人操作系统
- [User-in-the-Box](https://github.com/User-in-the-Box/user-in-the-box) - 基础模型参考

## 许可证

本项目基于Apache-2.0许可证发布。