# 基于 CARLA 仿真器的违规驾驶行为识别系统

##  项目简介

本项目使用 **CARLA 自动驾驶仿真器**生成真实的驾驶场景数据，通过计算机视觉和数据分析技术，自动识别驾驶过程中的违规行为。系统能够实时检测超速、闯红灯、压线等常见驾驶违规，并提供详细的分析报告和可视化结果。

###  核心功能

- 🚗 **超速检测**：实时监测车辆速度，识别超过道路限速的行为
- 🚦 **闯红灯检测**：检测红灯状态下通过路口的行为
- 📏 **压线检测**：识别车辆压线或驶入禁行区域（如路肩、自行车道）
- 📊 **数据可视化**：生成速度曲线图、违规统计图、行驶轨迹图
- 📈 **性能评估**：与真实标签对比，计算检测准确率
- 📝 **报告生成**：自动生成详细的检测报告（TXT格式）


## 🚀 项目要求

### 硬件要求
- CPU: Intel Core i5 或更高
- 内存: 4GB 或更高
- 硬盘: 500MB 可用空间（用于数据存储）

### 软件要求
- **操作系统**: Windows 10/11, Ubuntu 18.04+, macOS 10.15+
- **Python版本**: 3.8 - 3.13
- **CARLA版本**: 0.9.13（使用预生成数据，无需安装）

### Python依赖


## 📦 安装步骤

### 1. 克隆项目
```bash
git clone https://github.com/yourusername/driving_violation_detection.git
cd driving_violation_detection

创建虚拟环境（可选）
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate

安装依赖
pip install -r requirements.txt

生成数据
python data/download_real_carla_data.py

运行主程序
# 运行主程序
python main.py
