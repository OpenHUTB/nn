# Autonomous Vehicle Navigation Using Deep Learning
基于深度学习的自动驾驶汽车导航系统，支持 CARLA 仿真环境。

## 快速开始

### 环境要求
- Ubuntu 20.04
- CARLA 0.9.13
- Python 3.7
- ROS Noetic

## 📁 项目结构
```bash
text
├── main/              # 主程序目录
├── models/            # 预训练模型
├── test/              # 测试脚本
└── carla_ros_ws/      # ROS版本（可选）
```

### 安装依赖
```bash
# 创建Python环境
conda create -n carla-ros python=3.7
conda activate carla-ros
pip install -r requirements.txt
```

## 🏗️ ROS工作空间构建

### 1. 构建ROS包
```bash
cd carla_ros_ws
catkin_make
source devel/setup.bash
```

### 2. 安装ROS依赖
```bash
cd src/carla_autonomous/utils
./install.sh
```

## 🎮 运行方法

### 一键启动
```bash
cd carla_ros_ws/src/carla_autonomous/utils
./run_carla.sh
```

### 手动启动
```bash
# 终端1：启动CARLA
./CarlaUE4.sh

# 终端2：启动ROS节点
cd carla_ros_ws
source devel/setup.bash
roslaunch carla_autonomous carla_autonomous.launch

# 终端3：控制客户端
cd carla_ros_ws
source devel/setup.bash
python src/carla_autonomous/scripts/carla_control_client.py
```

## 📖 基础版本（无ROS）

1. 启动CARLA仿真器

```bash
./CarlaUE4.sh
```
2. 配置轨迹点


python main/get_location.py  # 获取当前坐标

编辑 main/config.py 中的 TRAJECTORIES 配置

```bash
# 运行主程序
python main/main.py
```

## ⚡ 快速测试
```bash
# 刹车测试
python test/test_braking.py

# 驾驶测试
python test/test_driving.py
```

## ⚠️ 注意事项
1. 必须先构建ROS工作空间：`catkin_make`
2. 每次使用前需要source：`source devel/setup.bash`
3. 确保CARLA版本为0.9.13

## 📖 参考
参考项目：varunpratap222/Autonomous-Vehicle-Navigation-Using-Deep-Learning