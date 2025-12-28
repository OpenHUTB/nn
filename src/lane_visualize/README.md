# AutoPilot System V3.0: Perception, Decision & Control

这是一个模拟 **ADAS (高级驾驶辅助系统)** 的完整原型项目。
V3.0 版本不仅实现了对车道和车辆的**感知**，还加入了**决策逻辑**（碰撞预警）、**模拟控制**（虚拟方向盘）以及**黑匣子取证**功能。

![Status](https://img.shields.io/badge/Status-Active-success)
![Version](https://img.shields.io/badge/Version-V3.0-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)

## ✨ V3.0 核心特性

### 1. 🧠 智能感知 (Perception)
* **车道保持**: 使用 OpenCV + 霍夫变换 + **EMA 平滑滤波**，稳定追踪车道线。
* **目标检测**: 集成 **YOLOv8** 模型，实时识别车辆、行人、卡车等障碍物。
* **跳帧优化**: 采用 "Skip-Frame" 策略，大幅降低 CPU 负载，实现流畅运行。

### 2. 🛡️ 决策与预警 (Decision Making)
* **FCW (前向碰撞预警)**: 实时计算前车距离（基于视觉占比），触发 **"BRAKE!"** 红色警报。
* **LDW (车道偏离预警)**: 监测车辆中心与车道中心的偏差。

### 3. 🧭 模拟控制 (Control Simulation)
* **虚拟方向盘**: 屏幕下方通过 **Steering Dashboard** 实时显示转向角度。
* **自动修正**: 根据车道曲率自动计算方向盘应转动的角度。

### 4. 📹 黑匣子取证 (Event Data Recorder)
* **自动抓拍**: 当系统判定有碰撞风险（出现 "BRAKE!"）时，自动保存当前画面。
* **证据留存**: 图片自动保存在 `events/` 目录下，文件名包含精确时间戳。

## 🛠️ 环境准备

### 1. 运行环境
* Python 3.8+
* 推荐配置: 具有独立显卡 (NVIDIA) 可获最佳性能，但在 CPU 环境下也能通过跳帧策略流畅运行。

### 2. 安装依赖
本项目已移除沉重的 TensorFlow 依赖，仅需轻量级库：
```bash
pip install -r requirements.txt