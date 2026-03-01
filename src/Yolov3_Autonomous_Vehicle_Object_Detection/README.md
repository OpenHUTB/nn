# Autonomous Vehicle Object Detection and Trajectory Planning

## 项目简介
本项目是一个基于自动驾驶场景的毕业设计项目。主要目的是利用 YOLOv3 目标检测算法与 CARLA 自动驾驶模拟器相结合，实现车辆对周围环境的感知（目标检测）以及基于环境反馈的轨迹规划。

## 核心目标
* [ ] 搭建 CARLA 自动驾驶模拟器环境
* [ ] 集成 YOLOv3 算法进行实时目标检测
* [ ] 根据检测到的障碍物/目标进行基础的轨迹规划
* [ ] 使用 TensorBoard 监控模型与系统的性能

## 技术栈 (初步规划)
* **Python 3.7+**
* **CARLA Simulator (0.9.11)**
* **OpenCV / PyTorch / TensorFlow** (用于 YOLOv3 部署与推理)
* **TensorBoard**

## 后续开发计划
1. 初始化项目结构与依赖清单。
2. 编写并测试 YOLOv3 的加载与推理模块。
3. 编写 CARLA 环境连接代码。
4. 将目标检测结果与 CARLA 车辆控制接口融合。
5. 完善轨迹规划逻辑并输出测试日志。

---
*注：本项目仍在持续开发和完善中。*