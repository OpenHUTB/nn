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
├── src/                # 核心源代码文件夹：存放环境控制与算法逻辑
├── models/             # 模型文件夹：存放 YOLOv3 权重与网络定义
└── utils/              # 辅助工具文件夹：存放通用工具函数
```

## 快速开始 (Quick Start)
为了确保项目顺利运行，请遵循以下部署步骤：

### 1. 环境准备
- 确保已安装 **CARLA 0.9.11** 或更高版本。
- 启动 CARLA 服务器：
  ```bash
  # Windows 系统下在 CARLA 根目录运行
  CarlaUE4.exe
  ```

### 2. 克隆项目与安装依赖
- 克隆本项目到本地：
  ```bash
  git clone https://github.com/Regret1103111/nn.git
  cd CARLA_AutoVision_Navigator
  ```
- 安装所需的 Python 库：
  ```bash
  pip install -r requirements.txt
  ```

### 3. 项目配置
- 在运行前，请检查 `config.py` 中的 `CARLA_HOST` 和 `CARLA_PORT` 是否与您的模拟器设置一致。

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
- [ ] 编写 CARLA 客户端连接与环境初始化基础脚本。
- [ ] 接入并调试 YOLOv3 视觉目标检测模块。
- [ ] 实现车辆 PID 控制与局部轨迹规划逻辑。
- [ ] 系统模块整合，在 CARLA 城市地图中进行综合运行测试。
- [ ] 代码重构、添加详细注释与性能优化。

## 声明
本项目为课程作业/学术研究项目，代码仅供学习与教育参考。
```
