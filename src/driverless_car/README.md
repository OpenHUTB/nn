# 无人机图像分类深度学习Demo（无硬件/仿真依赖）
该项目实现了**无人机图像分类**的深度学习简单方向demo，无需无人机硬件和仿真软件，通过公开数据集模拟无人机采集的图像数据，搭建轻量化卷积神经网络（CNN）完成分类任务，并提供数据显示和可视化界面，帮助新手理解无人机深度学习的基本流程。

## 项目特点
- 🚀 **无硬件依赖**：使用CIFAR-10公开数据集模拟无人机航拍图像，无需无人机硬件和仿真软件。
- 🧠 **轻量化模型**：搭建适用于无人机端的轻量化CNN，兼顾性能与算力消耗。
- 📊 **可视化界面**：包含数据集样本展示、训练过程实时可视化、预测结果展示三大可视化模块。
- 🐞 **兼容性修复**：解决Matplotlib中文字体缺失、PyCharm后端兼容等问题，可直接运行。

## 环境配置
### 1. 依赖库安装
项目基于Python和PyTorch实现，需安装以下依赖库：
```bash
pip install torch torchvision matplotlib numpy
无人机网格路径规划（强化学习 Q-Learning 实现）
这是一个基于强化学习（Q-Learning）和网格地图的无人机路径规划项目，实现了无人机在含障碍物的二维网格中自主避障并规划从起点到终点的最优路径，全程通过代码模拟（无需硬件），并提供了美观的可视化界面展示训练过程和路径结果。
项目介绍
核心功能
自定义 gymnasium 网格环境，包含起点、终点、随机障碍物
基于 Q-Learning 算法训练智能体，学习上下左右移动的最优策略
实时可视化无人机的移动路径、网格环境和训练奖励变化
优化的可视化界面，支持坐标标注、路径方向箭头、移动平均奖励曲线
技术栈
环境搭建：gymnasium（gym 的维护版）自定义网格环境
强化学习算法：Q-Learning（ε- 贪心策略平衡探索与利用）
可视化：matplotlib（绘制网格、路径、奖励曲线）
数值计算：numpy
安装依赖
1. 卸载旧版 gym（可选）
bash
运行
pip uninstall gym -y
2. 安装依赖库（使用国内镜像源加速）
bash
运行
# 基础依赖（推荐）
pip install gymnasium numpy matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple

# 若需要gymnasium完整版（包含额外环境，如Atari）
pip install gymnasium[all] -i https://pypi.tuna.tsinghua.edu.cn/simple