# CARLA AutoVision Navigator

## 项目简介
欢迎来到 **CARLA AutoVision Navigator** 项目！本项目是一个基于高度真实的 **CARLA 模拟器** 开发的自动驾驶仿真系统。项目旨在将实时的视觉目标检测与动态轨迹规划相结合，实现自动驾驶车辆在复杂交通环境下的智能导航。

通过引入深度学习算法（YOLOv3），自动驾驶主车（Ego-vehicle）能够实时感知周围环境，并在仿真环境中做出安全的行驶与避障决策。

## 项目目录结构
```text
CARLA_AutoVision_Navigator/
├── config.py           # 全局配置模块：统一管理服务器连接与传感器参数
├── requirements.txt    # 环境依赖：列出项目运行所需的第三方库
├── README.md           # 项目说明文档
├── .gitignore          # Git 忽略文件：排除冗余文件
├── src/                # 核心源代码文件夹
│   ├── __init__.py     # 模块初始化文件
│   └── carla_client.py # CARLA 客户端连接与主车管理模块 (New!)
├── models/             # 模型文件夹：存放 YOLOv3 权重与网络定义
└── utils/              # 辅助工具文件夹：存放通用工具函数
```

## 快速开始 (Quick Start)

### 1. 环境准备
- 确保已安装 **CARLA 0.9.11** 或更高版本。
- 启动 CARLA 服务器：
  ```bash
  CarlaUE4.exe
  ```

### 2. 克隆项目与安装依赖
- 克隆本项目并安装依赖：
  ```bash
  git clone https://github.com/Regret1103111/nn.git
  pip install -r requirements.txt
  ```

### 3. 运行基础连接测试
现在您可以测试与 CARLA 服务器的连接并观察主车生成：
```bash
python src/carla_client.py
```
*运行效果：程序将连接模拟器，在地图上生成一辆特斯拉 Model 3，并在 3 秒后自动销毁以清理环境。*

## 项目目标
- 与 CARLA 仿真服务器建立稳定连接，并在地图中生成自动驾驶主车。
- 实时获取主车上挂载的 RGB 摄像头传感器数据。
- 部署 YOLOv3 目标检测模型，精准识别环境中的行人、其他车辆等交通参与者。
- 开发轨迹规划与底层车辆控制算法，实现车辆的安全导航与自动避障。

## 开发计划 (Roadmap)
本项目采用逐步迭代的模式进行开发，当前的开发路线图如下：
- [x] 初始化项目仓库，编写项目 README 文档。
- [x] 提交 `.gitignore` 和环境依赖配置文件。
- [x] 建立全局配置模块 (`config.py`) 并规范项目目录结构。
- [x] 完善项目安装与运行指南 (Documentation Update)。
- [x] 实现 CARLA 客户端连接与主车管理逻辑功能。
- [ ] 接入并调试视觉传感器数据获取模块。
- [ ] 接入并调试 YOLOv3 视觉目标检测模块。
- [ ] 实现车辆 PID 控制与局部轨迹规划逻辑。
- [ ] 系统模块整合，在 CARLA 城市地图中进行综合运行测试。
- [ ] 代码重构、添加详细注释与性能优化。

## 声明
本项目为课程作业/学术研究项目，代码仅供学习与教育参考。
```
