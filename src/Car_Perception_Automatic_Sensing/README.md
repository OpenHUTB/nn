# Car_Perception_Automatic_Sensing
自动驾驶环境感知自动感知系统，支持车辆、行人、车道线等目标的检测与分割。

## 功能模块
- 数据加载与预处理（支持 KITTI/Waymo 数据集）
- 深度学习模型搭建（CNN/Transformer 骨干网络）
- 模型训练与验证
- 推理可视化与结果导出## 
- 目录结构
Car_Perception_Automatic_Sensing/
- ├── data/ # 数据集目录
- ├── model/ # 模型定义目录
- ├── loss/ # 损失函数目录
- ├── train/ # 训练脚本目录
- ├── infer/ # 推理脚本目录
- ├── utils/ # 工具类目录
- ├── test/ # 测试用例目录
- ├── requirements.txt # 依赖配置
- └── README.md # 项目文档


## 环境要求
Python >= 3.8，PyTorch >= 2.0.0

## 快速开始
1. 克隆仓库
2. 安装依赖
3. 准备数据集
4. 运行训练脚本