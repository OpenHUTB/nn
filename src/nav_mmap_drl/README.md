# 多模态机器人导航系统

## 项目概述

本项目旨在开发一套多模态机器人导航系统，该系统通过感知模块、跨域注意力模块和决策模块处理来自各类传感器（如惯性测量单元、摄像头和激光雷达）的数据，最终输出机器人在真实场景中的行动策略。该系统基于 CARLA 仿真器构建，并利用深度学习技术进行训练和测试。

## 目录

- [项目概述](#项目概述)
- [环境配置](#环境配置)
- [代码运行方式](#代码运行方式)
- [训练与测试](#训练与测试)
- [模型部署](#模型部署)

## 环境配置

运行代码前，请确保你的环境已配置以下依赖项：

- Python 3.7+
- PyTorch 1.7+
- CARLA 0.9.11
- 其他依赖项可参见requirements.txt

### 安装依赖项

首先，克隆项目仓库

```bash
git clone https://github.com/yourusername/robot_navigation_system.git
cd robot_navigation_system
```
然后，安装 Python 依赖项：

```bash
pip install -r requirements.txt
```
注：确保 CARLA 仿真器已安装且环境变量配置正确。你可参考 CARLA 官方文档进行安装和配置：https://carla.readthedocs.io/en/latest/build_linux/

## 代码运行方法
运行仿真程序你可通过以下命令运行集成系统并在 CARLA 仿真器中测试

```bash
python run_simulation.py
```
该脚本将启动 CARLA 仿真器，并执行完整的感知、注意力和决策模块，生成机器人的导航策略。

## 训练与测试
训练模型使用以下命令训练模型：
```bash
python main.py --mode train
```
该命令将从仿真数据集中加载数据，训练感知、跨域注意力和决策模块。训练参数及其他超参数可在main.py中配置。

测试模型
训练完成后，你可通过以下命令测试模型：

```bash
python main.py --mode test
```
该命令将加载测试数据集，评估模型在新数据上的性能，并输出测试损失。

## 模型部署
需注意的是，本智能车辆的相关底层控制配置将后续发布。此处仅预发布算法部署和推理的相关步骤。
基于 Jetson_robot 的模型训练与导出：推理前，需在 Jetson 机器人上训练模型并将其导出为 ONNX 格式：

```bash
python /_agent/_lightweight/train.py
```

加载模型并进行推理：

```bash
python _jetson_robot/deploy.py --onnx_model_path /_model/mmap_model.onnx