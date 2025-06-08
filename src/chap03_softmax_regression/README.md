# Logistic回归以及softmax回归


## 问题描述
1. 完成 `logistic_regression-exercise.ipnb`中的填空
   * 填空一：实现sigmoid的交叉熵损失函数(不使用tf内置的loss 函数)
   
1. 完成 `softmax_regression-exercise.ipnb`中的填空
    * 填空一：在__init__构造函数中建立模型所需的参数
    * 填空二：实现softmax的交叉熵损失函数(不使用tf内置的loss 函数)

# logistic_regression-exercise.py 项目说明

## 一、项目概述
本项目是一个使用 TensorFlow 实现逻辑回归的示例代码。通过从两个不同的高斯分布中采样生成数据集，然后使用逻辑回归模型对数据进行分类。项目展示了如何建立逻辑回归模型、定义损失函数、进行梯度下降训练，并最终可视化训练过程和结果。

## 二、代码结构
### 1. 数据集生成
- 从两个不同的高斯分布中采样生成正样本和负样本。
- 正样本 `C1` 从高斯分布 `N(3, 6, 1, 1, 0)` 采样，标签为 1。
- 负样本 `C2` 从高斯分布 `N(6, 3, 1, 1, 0)` 采样，标签为 0。
- 将正样本和负样本连接成一个数据集，并随机打乱顺序。

### 2. 模型建立
- **`LogisticRegression` 类**：定义逻辑回归模型，包含权重 `W` 和偏置 `b`，并应用 L2 正则化防止过拟合。
- **`compute_loss` 函数**：实现 sigmoid 的交叉熵损失函数（不使用 TensorFlow 内置的损失函数），并计算准确率。
- **`train_one_step` 函数**：执行一步梯度下降训练，更新模型的参数。

### 3. 模型训练
- 实例化逻辑回归模型和 Adam 优化器。
- 进行 200 次训练迭代，每 20 次迭代打印一次损失和准确率。

### 4. 结果展示
- 使用 `matplotlib` 库创建动画，可视化训练过程中拟合直线的变化和损失值的更新。

## 三、代码使用说明
### 1. 环境要求
- Python 3.x
- TensorFlow
- Matplotlib
- Numpy

### 2. 运行代码
将代码保存为 `logistic_regression-exercise.py`，在终端中运行以下命令：
```bash
python logistic_regression-exercise.py
```

### 3. 代码修改
- **数据集参数**：可以修改 `dot_num` 改变数据点的数量，修改高斯分布的均值和标准差来改变数据集的分布。
- **模型参数**：可以修改 `LogisticRegression` 类中的 L2 正则化系数和初始权重的范围，以及优化器的学习率。
- **训练迭代次数**：可以修改 `for` 循环中的迭代次数来改变训练的轮数。

## 四、注意事项
- 代码中使用了 `tf.function` 将部分函数编译为静态图，以提高执行效率。
- 在计算 sigmoid 交叉熵损失时，添加了一个极小值 `epsilon` 来防止对数运算出现数值不稳定问题。
- 代码最后使用 `HTML(anim.to_html5_video())` 显示动画，需要在 Jupyter Notebook 中运行才能正常显示。如果在终端中运行，可能需要将动画保存为视频文件。

## 五、总结
本项目通过一个简单的示例展示了如何使用 TensorFlow 实现逻辑回归模型，并进行训练和可视化。通过修改代码中的参数，可以进一步探索不同数据集和模型参数对分类结果的影响。

---

# softmax_regression-exercise.py项目说明

## 简介

本项目展示了如何使用 TensorFlow 实现软最大回归（Softmax Regression）模型。通过该模型，您可以对三类数据进行分类，并计算交叉熵损失和准确率。

## 文件结构

-   `softmax_regression-exercise.py`: 包含数据生成、模型定义和训练过程的主要代码。

## 环境要求

-   Python 3.x
-   TensorFlow 2.x
-   Matplotlib
-   NumPy

## 安装依赖

可以使用以下命令安装所需的 Python 库：

```bash
pip install tensorflow matplotlib numpy
```

## 使用说明

### 1. 数据生成

代码中生成三类数据点：

-   类别 1: 从高斯分布 `N(3, 6, 1, 1, 0)` 生成。
-   类别 2: 从高斯分布 `N(6, 3, 1, 1, 0)` 生成。
-   类别 3: 从高斯分布 `N(7, 7, 1, 1, 0)` 生成。

### 2. 模型定义

`SoftmaxRegression` 类定义了模型的参数和前向传播过程。您可以通过修改 `input_dim` 和 `num_classes` 参数来适应不同的输入特征和类别数量。

### 3. 训练模型

模型使用随机梯度下降（SGD）优化器进行训练。运行以下代码进行训练：

```python
for i in range(1000):
    loss, accuracy = train_one_step(model, opt, x, y)
    if i % 50 == 49:
        print(f'loss: {loss.numpy():.4}\t accuracy: {accuracy.numpy():.4}')
```

### 4. 结果展示

训练完成后，代码会绘制数据点和模型的决策边界。
