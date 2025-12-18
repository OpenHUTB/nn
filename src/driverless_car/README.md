
# 无人机深度学习项目集合

本项目集合包含两个无人机相关的深度学习Demo，均基于代码仿真实现，无需硬件依赖。

---

## 项目1：无人机图像分类深度学习Demo

### 项目简介
该项目实现了**无人机图像分类**的深度学习Demo，使用公开数据集模拟无人机采集的图像数据，搭建轻量化卷积神经网络（CNN）完成分类任务。

### 主要特点
- 🚀 **无硬件依赖**：使用CIFAR-10公开数据集模拟无人机航拍图像
- 🧠 **轻量化模型**：搭建适用于无人机端的轻量化CNN，兼顾性能与算力消耗
- 📊 **可视化界面**：包含数据集样本展示、训练过程实时可视化、预测结果展示
- 🐞 **兼容性修复**：解决Matplotlib中文字体缺失、PyCharm后端兼容等问题

### 技术栈
- Python + PyTorch
- torchvision (数据集处理)
- matplotlib (可视化)
- numpy (数值计算)

---

## 项目2：无人机网格路径规划（强化学习 Q-Learning 实现）

### 项目简介
基于强化学习（Q-Learning）和网格地图的无人机路径规划项目，实现了无人机在含障碍物的二维网格中自主避障并规划从起点到终点的最优路径。

### 核心功能
- 自定义 gymnasium 网格环境，包含起点、终点、随机障碍物
- 基于 Q-Learning 算法训练智能体，学习上下左右移动的最优策略
- 实时可视化无人机的移动路径、网格环境和训练奖励变化
- 优化的可视化界面，支持坐标标注、路径方向箭头、移动平均奖励曲线

### 技术栈
- **环境搭建**：gymnasium（gym 的维护版）自定义网格环境
- **强化学习算法**：Q-Learning（ε-贪心策略平衡探索与利用）
- **可视化**：matplotlib（绘制网格、路径、奖励曲线）
- **数值计算**：numpy

---

## 环境配置

### 基础依赖安装

推荐使用国内镜像源加速安装：

```bash
# 基础依赖（推荐）
pip install torch torchvision matplotlib numpy gymnasium -i https://pypi.tuna.tsinghua.edu.cn/simple

# 若需要gymnasium完整版（包含额外环境，如Atari）
pip install gymnasium[all] -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 卸载旧版依赖（可选）
```bash
# 卸载旧版 gym（如果存在冲突）
pip uninstall gym -y
```

### 系统要求
- Python 3.7+
- 支持CUDA的GPU（可选，用于加速训练）

---

## 项目结构

```
无人机深度学习项目/
├── README.md                    # 项目说明文档
├── 图像分类项目/                  # 无人机图像分类Demo
│   ├── main.py                 # 主程序入口
│   ├── model.py                # CNN模型定义
│   ├── train.py                # 训练脚本
│   ├── visualize.py            # 可视化工具
│   └── utils.py                # 工具函数
└── 路径规划项目/                  # 无人机路径规划Demo
    ├── main.py                 # 主程序入口
    ├── environment.py          # 网格环境定义
    ├── q_learning.py           # Q-Learning算法实现
    ├── visualization.py        # 可视化工具
    └── utils.py                # 工具函数
```

---

## 使用说明

### 图像分类项目
1. 运行主程序开始训练和测试
2. 查看数据集样本展示
3. 观察训练过程可视化
4. 测试模型预测效果

### 路径规划项目
1. 启动网格环境
2. 运行Q-Learning训练过程
3. 观察路径规划结果
4. 查看训练奖励变化曲线

---

## 注意事项

- 项目完全基于代码仿真，无需真实无人机硬件
- 可在普通PC上运行，建议配置支持CUDA的GPU以加速训练
- 所有可视化界面均已优化，兼容中文显示
- 代码已处理常见的兼容性问题，可直接在PyCharm等IDE中运行

---

## 贡献指南

欢迎提交Issue和Pull Request来改进项目！

---

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

---

