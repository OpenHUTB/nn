# Active Lane Keeping Assistant

本项目实现了一个基于计算机视觉的主动车道保持系统。通过图像处理技术识别车道线，结合 PID 控制器实现车辆自动跟踪控制。项目使用 CARLA 模拟器进行测试和验证。

## 目录
- [项目简介](#introduction)
- [安装步骤](#installation)
- [快速开始](#quick-start)
- [核心技术](#core-technology)
- [参考资料](#references)

## 项目简介 <a name="introduction"></a>

主动车道保持助手旨在通过计算机视觉技术实现车辆的自动车道保持功能。主要特点：

- **图像识别**：使用 OpenCV 进行车道线检测，无需机器学习模型
- **控制算法**：支持多种控制器（简单转向、P、PD、PID）
- **仿真测试**：基于 CARLA 模拟器进行真实环境模拟

## 安装步骤 <a name="installation"></a>

### 环境要求
- Python 3.10
- CARLA 模拟器 0.9.13
- Windows/Linux/macOS

### 安装流程
1. 克隆项目：
```sh
git clone https://github.com/C2G-BR/Active-Lane-Keeping-Assistant.git
cd Active-Lane-Keeping-Assistant
```

2. 下载并安装 [CARLA 0.9.13](https://github.com/carla-simulator/carla/releases/tag/0.9.13)

3. 安装 Python 依赖：
```sh
pip install -r requirements.txt
```

## 快速开始 <a name="quick-start"></a>

### 运行程序
1. 启动 CARLA 模拟器
2. 在项目目录中运行：
```sh
python src/main.py -id "test" -c pid -s 1000
```

### 命令参数
```sh
python main.py -h
```
- `-id`：运行标识名称
- `-c`：控制器类型（simple/p/pd/pid）
- `-s`：运行步数

### 预期效果
启动后 CARLA 模拟器中会出现一辆自动行驶的车辆，车辆会自动识别车道并保持在车道中心行驶。

## 核心技术 <a name="core-technology"></a>

### 计算机视觉车道识别

采用传统图像处理技术检测车道线：

1. **图像预处理**：颜色空间转换、高斯模糊、二值化
2. **感兴趣区域提取**：只处理道路区域，减少计算量
3. **透视变换**：转换为鸟瞰视角，便于车道识别
4. **直方图分析**：定位车道边界位置
5. **滑动窗口法**：拟合车道曲线

### PID 控制器

支持多种控制算法对比：

| 控制器 | 特点 |
| --- | --- |
| Simple | 固定转向角度，简单但抖动大 |
| P | 比例控制，平滑转向 |
| PD | 比例+微分，减少超调 |
| PID | 比例+积分+微分，最优控制 |

**优化后参数**：
- PID 控制器：P=0.65, I=0.00000002, D=0.034

## 参考资料 <a name="references"></a>

- [Lane Detection with Deep Learning](https://towardsdatascience.com/lane-detection-with-deep-learning-part-1-9e096f3320b7)
- [Real-time Lane Detection with OpenCV](https://www.analyticsvidhya.com/blog/2020/05/tutorial-real-time-lane-detection-opencv/)
- [CARLA Simulator Documentation](https://carla.readthedocs.io/)

---

*本项目由 [@Irish-77](https://github.com/Irish-77) 和 [@Ronho](https://github.com/Ronho) 合作完成*