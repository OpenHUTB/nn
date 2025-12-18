# 环境设置
运行代码前，请确保环境满足以下依赖要求：

| 依赖项       | 版本要求       |
|--------------|----------------|
| Python       | 3.7+           |
| PyTorch      | 1.7+           |
| CARLA        | 0.9.11         |
| 其他依赖     | 详见 requirements.txt |

## 安装依赖
### 1. 克隆项目仓库
```bash
git clone https://github.com/yourusername/robot_navigation_system.git
cd robot_navigation_system
```

### 2. 安装Python依赖
```bash
pip install -r requirements.txt
```

> **注意**：需提前安装CARLA模拟器并正确配置环境变量，安装与配置指南参考CARLA官方文档：  
> https://carla.readthedocs.io/en/latest/build_linux/

# 如何运行代码
## 运行模拟
执行以下命令启动集成系统，并在CARLA模拟器中完成测试：
```bash
python run_simulation.py
```
该脚本将自动启动CARLA模拟器，加载感知、注意力与决策模块，生成机器人导航策略。

## 培训与测试
### 训练模型
```bash
python main.py --mode train
```
- 功能：加载模拟数据集，训练感知、跨域注意力及决策模块
- 配置：训练参数、超参数等可在 `main.py` 文件中调整

### 测试模型
训练完成后，执行以下命令测试模型：
```bash
python main.py --mode test
```
- 功能：加载测试数据集，评估模型在新数据上的表现
- 输出：测试损失值

# 模型部署
> 说明：智能车辆底层控制配置将后续发布，当前仅提供算法部署与推理相关步骤。

### 基于Jetson_robot的模型训练与推理
#### 1. 训练模型并导出为ONNX格式（适配Jetson机器人）
```bash
python /_agent/_lightweight/train.py
```

#### 2. 加载ONNX模型并执行推理
```bash
python _jetson_robot/deploy.py --onnx_model_path /_model/mmap_model.onnx
```