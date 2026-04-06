# 机械臂控制与仿真项目
A Robotic Arm Control and Simulation Project

## 项目简介
本项目旨在实现对六轴工业机械臂的运动控制、路径规划与仿真验证，支持离线编程、实时姿态解算和简单的抓取任务调度。项目采用模块化设计，可快速适配不同型号的机械臂硬件，同时提供可视化仿真界面，降低开发与调试成本。

## 核心功能
- 🎯 正逆运动学解算：基于D-H参数法实现机械臂关节角与末端位姿的相互转换
- 📍 路径规划：支持直线插补、圆弧插补、关节空间轨迹规划
- 🎮 控制模式：提供手动示教、自动运行、仿真验证三种控制模式
- 📊 状态监控：实时显示关节角度、末端坐标、执行速度等关键参数
- 🔌 硬件适配：支持串口/以太网与机械臂控制器通信（兼容主流工业协议）

## 技术栈
| 模块         | 技术/工具                          |
|--------------|------------------------------------|
| 核心算法     | Python (NumPy, SciPy)              |
| 运动学解算   | PyKDL / 自定义D-H解算库            |
| 仿真可视化   | PyQt5 / RViz (可选)                |
| 硬件通信     | pyserial / socket                  |
| 依赖管理     | pip / requirements.txt             |

## 快速开始
### 环境要求
- Python 3.8+
- Windows/Linux/macOS (Linux推荐，兼容性最佳)
- 机械臂控制器（如支持Modbus/TCP的控制器，或仿真模式无需硬件）

### 安装依赖
```bash
# 克隆项目（替换为你的仓库地址）
git clone https://github.com/your-username/robotic-arm-project.git
cd robotic-arm-project

# 安装依赖包
pip install -r requirements.txt