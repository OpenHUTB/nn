# Mojoco DataSim

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 项目运行环境
### 系统要求
- Windows 10/11 (64位) / Ubuntu 18.04+/CentOS 7+ (Linux) / macOS 12+
- 内存：至少8GB（推荐16GB及以上，处理大规模点云/图像数据时需32GB+）
- GPU：NVIDIA GPU（显存6GB+，推荐10GB+，支持CUDA 11.0+，可选，用于加速数据生成）

### 依赖环境与库
- Python 3.8 / 3.9 / 3.10（推荐3.9）
- 核心依赖库：numpy>=1.21.0
- opencv-python>=4.5.5
- pyquaternion>=0.9.9
- pandas>=1.4.0
- matplotlib>=3.5.0

### 项目介绍
Mojoco DataSim 是一款基于 MuJoCo 物理仿真引擎开发的自动驾驶仿真数据生成工具，专注于为自动驾驶算法训练提供带物理属性的高精度标注数据。
与传统仿真数据工具不同，本项目依托 MuJoCo 的多体动力学仿真能力，让生成的车辆运动、传感器数据更贴近真实物理世界，解决了纯视觉仿真数据 “物理失真”、与实车数据偏差大的问题。
核心特性

MuJoCo 原生物理仿真：基于 MuJoCo 的 XML 模型定义（车辆、道路、交通参与者），实现车辆的动力学运动（转向、制动、悬挂响应）、交通参与者的物理交互（碰撞、超车、变道）仿真，数据具备真实物理属性。
多传感器数据生成：同步生成激光雷达（LiDAR）点云、车载摄像头图像、毫米波雷达数据、IMU/GPS 数据，传感器模型与 MuJoCo 的物理场景深度绑定。
全自动高精度标注：利用 MuJoCo 的场景语义信息（物体 ID、位置、姿态），自动为数据添加目标检测、3D 框、语义分割、实例分割标注，标注格式兼容 KITTI、YOLO、COCO 等主流数据集。
轻量化场景定制：通过修改 MuJoCo 的 XML 配置文件，即可自定义车辆参数（轴距、质量、动力）、场景环境（城市道路、高速、园区）、天气（雨天、雾天）和传感器参数（激光雷达线束、摄像头分辨率）。
核心应用场景

自动驾驶感知算法（3D 目标检测、语义分割）的训练与验证，尤其适合对物理真实性要求高的算法。
车辆控制算法（路径跟踪、避障）的仿真数据生成与测试。
极端交通场景（如车辆碰撞、紧急制动）的数据生成，弥补真实数据采集的安全风险与成本问题。

### 项目发展方向

短期方向（V1.0-V1.5）
MuJoCo 模型库扩充：完善主流车型（轿车、SUV、货车）的 MuJoCo XML 模型库，提供可直接复用的车辆动力学参数。
传感器模型优化：基于 MuJoCo 的光线追踪能力，提升激光雷达点云的反射率仿真、摄像头的光照 / 阴影仿真效果。
标注格式扩展：增加对 nuScenes、Waymo 等自动驾驶顶级数据集格式的支持，便于与真实数据融合训练。
中期方向（V2.0-V2.5）

场景自动化生成：结合算法自动生成多样化的 MuJoCo 交通场景（如随机车辆分布、行人横穿路径），支持批量场景数据生成。
MuJoCo 与 AI 融合：引入强化学习（RL），基于 MuJoCo 的仿真环境自动生成算法鲁棒性测试所需的边缘场景数据。
实时渲染优化：基于 MuJoCo 的 GPU 渲染能力，提升大规模场景下的数据生成速度（如实时输出高清图像和点云）。
长期方向（V3.0+）

MuJoCo 生态集成：支持与 MuJoCo 的官方工具链（如 MuJoCo MPC、MuJoCo Physics）对接，实现从数据生成到控制算法测试的全流程闭环。
跨平台兼容：支持将 MuJoCo 的仿真场景与数据导出为 ROS/ROS2 话题，与 Autoware、Apollo 等自动驾驶框架无缝集成。
开源社区建设：开放用户自定义的 MuJoCo 车辆 / 场景模型库，吸引开发者贡献模型和场景配置，形成基于 MuJoCo 的自动驾驶数据生成生态

快速开始
1. 克隆项目
bash
运行
git clone https://github.com/your-username/mojoco-datasim.git
cd mojoco-datasim
2. 安装依赖
bash
运行
# 1. 安装MuJoCo（可跳过，pip安装mujoco库会自动包含）
# 参考：https://mujoco.org/download

# 2. 安装Python依赖
pip install -r requirements.txt

# 3.运行示例
bash
运行
# 运行单一场景数据生成示例（基于MuJoCo的城市道路场景）
python examples/generate_mujoco_scene.py
# 查看生成的数据与标注（默认存储在output/目录下）
python examples/visualize_data.py