# 机器人导航系统 - 环境配置与运行手册
## 一、环境设置
运行本项目代码前，需确保开发环境满足以下依赖版本要求：
- Python 3.7 及以上版本
- PyTorch 1.7 及以上版本
- CARLA 模拟器 0.9.11 版本
- 其余 Python 依赖项详见项目根目录下的 `requirements.txt` 文件

## 二、依赖安装步骤
### 1. 克隆项目仓库
```bash
git clone https://github.com/yourusername/robot_navigation_system.git
cd robot_navigation_system
```

### 2. 安装Python依赖
```bash
pip install -r requirements.txt
```

> **重要提示**：需提前完成CARLA模拟器的安装，并正确配置相关环境变量。CARLA安装与配置可参考官方文档：[https://carla.readthedocs.io/en/latest/build_linux/](https://carla.readthedocs.io/en/latest/build_linux/)

## 三、代码运行方法
### 运行系统模拟
执行以下命令可启动CARLA模拟器，加载完整的感知、注意力与决策模块，生成机器人导航策略：
```bash
python run_simulation.py
```

## 四、模型训练与测试
### 1. 训练模型
执行以下命令加载模拟数据集，训练感知、跨域注意力和决策模块：
```bash
python main.py --mode train
```
> 说明：训练参数、超参数等可在 `main.py` 文件中按需配置。

### 2. 测试模型
模型训练完成后，执行以下命令加载测试数据集，评估模型在新数据上的表现并输出测试损失：
```bash
python main.py --mode test
```

## 五、模型部署
> 说明：智能车辆底层控制配置将后续发布，以下仅提供算法部署与推理的核心步骤。

### 1. Jetson_robot 模型训练与导出
推理前需先训练模型，并将其导出为Jetson机器人兼容的ONNX格式：
```bash
python /_agent/_lightweight/train.py
```

### 2. 加载模型并推理
```bash
python _jetson_robot/deploy.py --onnx_model_path /_model/mmap_model.onnx
```

### 总结
1. 环境核心要求：Python 3.7+、PyTorch 1.7+、CARLA 0.9.11，需确保CARLA环境变量配置正确；
2. 核心操作命令：模拟运行用 `run_simulation.py`，模型训练/测试用 `main.py --mode [train/test]`；
3. Jetson部署关键步骤：先训练导出ONNX模型，再通过 `deploy.py` 指定模型路径加载推理。