# 自动驾驶系统中的对抗性横穿马路行为建模

本研究聚焦于在Carla模拟环境中对自动驾驶汽车进行测试时的真实对抗性行人行为建模。所有方法均在Carla仿真平台中实现。

## 核心创新：基于社会力与机器学习的混合行为建模方法

本研究的核心创新在于提出了一种**新颖的混合建模方法**用于生成式行人行为建模，该方法能够结合**社会力模型**和**机器学习模型**各自的优势，相互补偿彼此的局限性。社会力模型具有高度交互性和对未见场景的适应性，而机器学习模型则能够捕捉高方差的行为模式。两者结合，可以有效生成丰富且对未见情境具有鲁棒性的生成式行为模型。

### 创新动机

我们开展这项工作的动机在于：我们既有可用于训练机器学习模型的可用数据，也有无法直接使用的不可用数据。因此，我们创新性地提出新方法，创建能够从不同类型数据中受益的混合模型，充分利用各类数据资源。

### 技术路线

本研究提出的混合建模方法包含以下关键技术路线：

- **程序生成与深度生成模型结合**：利用程序生成、变分自编码器、生成对抗网络和Transformer等技术原型设计生成式行为模型

- **软轨迹建模**：将基于规划的方法与微观行为模型相结合的生成式行人模型，解决从机器学习模型向分布外场景重新定向计划的问题

- **微观与宏观行为融合**：同时发现高层决策行为和低层微观运动与机动，实现跨层级的行为建模

- **对抗性测试**：利用对抗性技术高效识别自动驾驶系统的失败案例

### 相关论文

```bibtex
@inproceedings{inproceedings,
author = {Muktadir, Golam Md and Whitehead, Jim},
journal={2024 IEEE International Conference on Robotics and Automation (ICRA)},
year = {2024},
month = {05},
title = {Adaptive Pedestrian Agent Modeling for Scenario-based Testing of Autonomous Vehicles through Behavior Retargeting}
}

@article{Muktadir2022AdversarialJM,
  title={Adversarial jaywalker modeling for simulation-based testing of Autonomous Vehicle Systems},
  author={Golam Md Muktadir and E. James Whitehead},
  journal={2022 IEEE Intelligent Vehicles Symposium (IV)},
  year={2022},
  pages={1697-1702},
}
 
