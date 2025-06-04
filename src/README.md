
## 1. 全连接神经网络

### 问题描述
利用 `numpy`、`tensorflow` 和 `pytorch` 搭建全连接神经网络。其中，使用 `numpy` 实现时需要手动求导，而 `tensorflow` 和 `pytorch` 具备自动求导机制。

### 数据集
MNIST 数据集包含 60000 张训练图片和 10000 张测试图片。该数据集样本数量充足，可用于训练复杂模型（如 CNN 深层神经网络），常作为新模式识别模型的测试用例，且数据规模相对较小，可在笔记本 CPU 上直接运行。

### 题目要求
补全本章节中所有 `.ipynb` 文件中提示补全的部分。

### 代码实现建议
- 对于使用 `numpy` 实现的部分，需要仔细推导和实现各层的前向传播和反向传播过程。
- 使用 `tensorflow` 或 `pytorch` 时，可利用其自动求导功能简化代码编写。

### 参考资料
可参考相关深度学习教材和在线教程，深入理解全连接神经网络的原理和实现方法。

## 2. 函数拟合

### 问题描述
理论和实验表明，一个两层的 ReLU 网络可以模拟任何函数 [1 - 5]。需要自行定义一个函数，并使用基于 ReLU 的神经网络来拟合此函数。

### 要求
- **数据采样**：自行在函数上采样生成训练集和测试集，使用训练集训练神经网络，测试集验证拟合效果。
- **框架选择**：可以使用深度学习框架（如 `tensorflow`、`pytorch`、`keras` 等）编写模型。若不使用上述框架，直接用 `NumPy` 实现可获得最高 5 分的附加分。
- **提交内容**：提交时需一并提交代码和报告。
  - **代码注释**：代码建议注释清楚（5 分）。
  - **报告内容**：报告至少应包含函数定义、数据采集、模型描述、拟合效果等部分（5 分）。

### 示例
![函数拟合示例](fitting.jpg)

### 参考文献
[1] G. Cybenko. 1989. Approximation by superpositions of a sigmoidal function.
[2] K. Hornik, M. Stinchcombe, and H. White. 1989. Multilayer feedforward networks are universal approximators.
[3] Moshe Leshno, et al. 1993. Multilayer feedforward networks with a non - polynomial activation function can approximate any function
[4] Vinod Nair and Geoffrey E. Hinton. 2010. Rectified linear units improve restricted boltzmann machines.
[5] Xavier Glorot, Antoine Bordes, Yoshua Bengio. 2011. Deep Sparse Rectifier Neural Networks. PMLR 15:315 - 323.

### 代码实现建议
- 在选择函数时，可考虑不同复杂度的函数，如线性函数、非线性函数等。
- 对于数据采样，要确保训练集和测试集的分布合理。
- 编写报告时，详细描述每个步骤和结果，可使用图表辅助说明。

## TensorFlow 2.0 练习说明

### 简介
本项目实现了一些基本的 TensorFlow 2.0 练习，包括自定义的 `softmax` 函数、`sigmoid` 函数以及它们各自的交叉熵损失函数。通过手动计算这些常用函数，展示了如何在不依赖 TensorFlow 内置函数的情况下实现相关功能。

### 文件结构
- `tf2.0-exercise.py`：包含上述功能实现的主要代码。

### 环境要求
- Python 3.x
- TensorFlow 2.x
- NumPy

### 安装依赖
可以使用以下命令安装所需的 Python 库：
```bash
pip install tensorflow numpy
```

### 使用说明

#### 1. Softmax 函数
`softmax(x)` 函数实现了数值稳定的 `softmax` 操作，适用于任意形状的输入张量。调用示例：
```python
import tensorflow as tf
from tf2.0-exercise import softmax

result = softmax(tf.constant([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))
```

#### 2. Sigmoid 函数
`sigmoid(x)` 函数实现了 `sigmoid` 操作，返回输入的 `sigmoid` 概率。调用示例：
```python
import tensorflow as tf
from tf2.0-exercise import sigmoid

result = sigmoid(tf.constant([[0.0], [1.0], [2.0]]))
```

#### 3. Softmax 交叉熵损失函数
`softmax_ce(logits, label)` 函数计算 `softmax` 交叉熵损失。该函数接受未经 `softmax` 处理的原始输出（logits）和 one - hot 格式的标签。调用示例：
```python
import tensorflow as tf
from tf2.0-exercise import softmax_ce

loss = softmax_ce(logits=tf.constant([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]), label=tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
```

#### 4. Sigmoid 交叉熵损失函数
`sigmoid_ce(x, label)` 函数计算 `sigmoid` 交叉熵损失。调用示例：
```python
import tensorflow as tf
from tf2.0-exercise import sigmoid_ce

loss = sigmoid_ce(x=tf.constant([0.5, 0.8]), label=tf.constant([1.0, 0.0]))
```

### 测试
代码中包含测试用例，用于验证自定义函数的正确性。通过比较自定义函数与 TensorFlow 内置函数的结果，确保实现的准确性。

### 代码实现建议
- 在实现自定义函数时，要注意数值稳定性问题，避免出现溢出或下溢情况。
- 编写测试用例时，要覆盖不同的输入情况，确保函数的正确性。

## MNIST FNN with Numpy

### 项目简介
该项目实现了一个简单的前馈神经网络（Feedforward Neural Network, FNN），用于识别手写数字。网络使用 `Numpy` 实现，并通过自动微分计算梯度。数据集使用的是经典的 MNIST 数据集。

### 目录结构
```
.
├── tutorial_minst_fnn-numpy-exercise.py  # 主代码文件
```

### 环境要求
- Python 3.x
- Numpy
- TensorFlow

### 安装依赖
使用以下命令安装所需的 Python 包：
```bash
pip install numpy tensorflow
```

### 使用方法

#### 1. 准备数据
通过调用 `mnist_dataset()` 函数加载并预处理 MNIST 数据集。
```python
from tutorial_minst_fnn-numpy-exercise import mnist_dataset

train_data, train_label, test_data, test_label = mnist_dataset()
```

#### 2. 定义模型
创建 `myModel` 类，包含前向传播和反向传播的方法。
```python
from tutorial_minst_fnn-numpy-exercise import myModel

model = myModel()
```

#### 3. 训练模型
使用 `train()` 函数训练模型，训练过程中会打印每个 epoch 的损失和准确率。
```python
from tutorial_minst_fnn-numpy-exercise import train

losses, accuracies = train(model, train_data, train_label)
```

#### 4. 测试模型
使用 `test()` 函数在测试集上评估模型性能。
```python
from tutorial_minst_fnn-numpy-exercise import test

test_loss, test_accuracy = test(model, test_data, test_label)
print(f'Test Loss {test_loss:.4f}; Test Accuracy {test_accuracy:.4f}')
```

### 代码说明
- **Matmul**：实现矩阵乘法及其梯度计算。
- **Relu**：实现 ReLU 激活函数及其梯度计算。
- **Softmax**：实现 Softmax 函数及其梯度计算。
- **Log**：实现 Log Softmax 函数及其梯度计算。
- **compute_loss**：计算损失函数。
- **compute_accuracy**：计算模型准确率。

### 代码实现建议
- 在实现梯度计算时，要仔细推导公式，确保计算的正确性。
- 训练过程中可以调整学习率、批次大小等超参数，以提高模型性能。

## MNIST 手写数字识别教程

### 简介
本项目使用 TensorFlow 2.0 实现一个简单的前馈神经网络（Feedforward Neural Network, FNN），用于识别 MNIST 数据集中的手写数字。该项目主要包括数据准备、模型定义、训练和测试过程。

### 环境要求
- Python 3.x
- TensorFlow 2.0+
- NumPy

### 文件结构
- `tutorial_minst_fnn-tf2.0-exercise.py`：主程序，包含数据加载、模型定义、训练和测试逻辑。

### 使用说明

#### 1. 准备数据
程序首先加载 MNIST 数据集，并对图像数据进行归一化处理，将像素值缩放到 [0, 1] 之间。
```python
from tutorial_minst_fnn-tf2.0-exercise import mnist_dataset

train_data, train_label, test_data, test_label = mnist_dataset()
```

#### 2. 定义模型
创建了一个简单的两层神经网络模型，包含：
- 输入层：784 个节点（28x28 像素展平）
- 隐藏层：128 个 ReLU 激活的节点
- 输出层：10 个节点（数字 0 - 9）
```python
from tutorial_minst_fnn-tf2.0-exercise import myModel

model = myModel()
```

#### 3. 训练模型
使用 Adam 优化器进行模型训练，执行 50 个 epoch。每个 epoch 将输出当前的损失和准确率。
```python
from tutorial_minst_fnn-tf2.0-exercise import train

losses, accuracies = train(model, train_data, train_label)
```

#### 4. 测试模型
在测试集上评估训练后的模型，输出测试损失和准确率。
```python
from tutorial_minst_fnn-tf2.0-exercise import test

test_loss, test_accuracy = test(model, test_data, test_label)
print(f'Test Loss {test_loss:.4f}; Test Accuracy {test_accuracy:.4f}')
```

### 如何运行
确保安装了所需的库后，可以通过以下命令运行该程序：
```bash
python tutorial_minst_fnn-tf2.0-exercise.py
```

### 代码说明
- **数据加载**：通过 `mnist_dataset()` 函数加载数据。
- **模型定义**：使用 `myModel` 类定义模型的结构和前向传播逻辑。
- **损失和准确率计算**：通过 `compute_loss()` 和 `compute_accuracy()` 函数计算损失和准确率。
- **训练步骤**：`train_one_step()` 函数执行一次训练步骤并更新模型参数。
- **测试步骤**：`test()` 函数计算测试集上的损失和准确率。

### 代码实现建议
- 可以尝试调整模型的结构，如增加隐藏层的数量或节点数，以提高模型性能。
- 在训练过程中，可以使用早停策略，避免过拟合。