# 自动驾驶车辆语义分割
<p align="center">
<img width="500px" src="examples/orig_labels_1.gif"/>
</p>

本项目为伍斯特理工学院（WPI）研究生课程**RBE549 计算机视觉**课程作业，团队以**语义分割**作为研究课题。项目基于CARLA自动驾驶仿真平台，通过自动化程序批量采集数千张带标注图像，构建语义分割专用数据集。本仓库完整收录该项目的全部代码、训练模型、效果示例及项目报告文档。

<p align="center">
<img width="500px" src="examples/movie2.gif"/>
</p>

最终，本研究采用**U-Net网络模型**，结合**稀疏分类交叉熵焦点损失函数**开展训练，以此缓解各类别样本数量严重失衡的问题，优化数据集分布不均衡带来的模型偏差。

团队在超参数调试、损失函数对比实验过程中，各模型的性能表现数据，详见项目报告及下方图表：

<p align="center">
<img width="500px" src="performance_graph.png">
</p>
<p align="center">
<img width="500px" src="performance_chart.png">
</p>

---
### 专业术语对照（便于学术使用）
1. Semantic Segmentation：语义分割
2. CARLA：卡拉自动驾驶仿真平台（通用专有名词不译）
3. U-Net：U型网络（经典分割模型，保留原名）
4. sparse categorical cross entropy focal loss：稀疏分类交叉熵焦点损失
5. class imbalance：类别不平衡
6. hyperparameter：超参数