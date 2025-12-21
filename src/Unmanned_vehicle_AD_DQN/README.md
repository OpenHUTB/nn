# 基于DQN实现自动驾驶模型

这是深度 Q 网络 （DQN） 算法的实现，用于在 CARLA 中训练自动驾驶模型

## 环境配置

1. python版本3.7.16
2. 确保将carla在运行前配置到系统环境中
3.environment.yml为conda导出的虚拟环境，可一键导入虚拟环境
4.ROS封装环境为Ubuntu20.04-Linux（ros版本是1版，不是ros2）

### carla环境配置

1.Carla版本尽量选择适配Python3，例如0.9.12版本
2. .whl文件的API请下载至py环境中

## 使用步骤

确保carla在代码运行之前已经启动

### 训练
训练模型请运行：

```

Main.py

```
### 测试
测试训练后的模型，请运行：
```

Test.py

```
### 文件说明
各文件说明：
```

main.py：训练主流程和循环控制，管理模型保存和训练策略

Model.py：定义神经网络结构和强化学习智能体，包含核心算法

Environment.py：Carla仿真环境，处理图像、行人、车辆交互和基础奖励

Test.py：模型评估和测试，加载训练好的模型进行性能验证

Hyperparameters.py：所有可调整的训练参数和配置


```
### 运行产生文件说明：
```

Log文件夹：文件夹里的文件是TensorBoard的记录文件，主要作用是可视化、分析和回顾训练过程，删除它们不会影响已经训练好的模型或中断正在进行的训练

models：这些是训练好的神经网络权重，是核心成果，请谨慎删除

```
### ROS封装
ROS文件夹：
```

launch\train_full.launch  为完整训练启动文件
scripts\carla_train_full.py  为完整训练节点
./start_full_training.sh  为一键启动脚本
CMakeLists.txt  为ROS编译配置
package.xml  为ROS包描述

```
### ROS代码启用
使用一键启动脚本/手动（此处不给出）：
```

（请确保Carla已启用）
chmod +x start_full_training.sh
./start_full_training.sh

```

