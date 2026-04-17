# CARLA 仿真 2D 跟踪：多算法对比与课程实践

## 1. 项目选题
本项目为机器学习课程大作业，旨在自动驾驶仿真平台 CARLA 中实现 2D 多目标跟踪算法，并对比 SORT、DeepSORT、OC-SORT 等经典跟踪器的性能差异。

## 2. 项目目标
- 在 CARLA 环境中实现车辆、行人的 2D 目标跟踪
- 对比不同多目标跟踪算法的效果与鲁棒性
- 完成环境搭建、代码调试、实验分析
- 有余力时扩展至 3D 目标跟踪，更贴近真实自动驾驶感知

## 3. 运行环境
- Python 3.9
- CARLA 0.9.14
- Windows 11
- 主要依赖：numpy, opencv-python, filterpy, lap, torch

## 4. 实现计划
1. 完成 CARLA 环境配置与调试
2. 实现基于真值的 2D 目标检测与跟踪
3. 集成 YOLO 检测器 + SORT/DeepSORT 跟踪器
4. 对比不同算法在不同场景下的性能
5. 结果可视化与分析总结
6. 可选扩展：3D 目标跟踪实现与对比

## 5. 参考与致谢
本项目基于优秀开源项目学习与复现：
- 基础框架参考：wuhanstudio/2d-carla-tracking
  https://github.com/wuhanstudio/2d-carla-tracking

在此向原作者表示感谢，本项目在此基础上完成课程实践、算法对比与扩展尝试。