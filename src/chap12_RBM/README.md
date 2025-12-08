# 受限玻尔兹曼机



## 问题描述：

​	使用受限玻尔兹曼机（Restricted Boltzmann Machine, RBM），对MNIST 数据集建模。其中，每个
像素点（值为0 或1）作为可观测变量，同时自定义一组隐变量（隐变量的个数为超参），完成以下任务：

1. 使用不带标签的MNIST 样本数据，训练RBM 参数。
2. 根据第1 步训练出的RBM，使用Gibbs 采样的方法，生成一组服从分布的样本。






## 数据集: 

 	MNIST数据集包括60000张训练图片和10000张测试图片。图片样本的数量已经足够训练一个很复杂的模型（例如 CNN的深层神经网络）。它经常被用来作为一个新 的模式识别模型的测试用例。而且它也是一个方便学生和研究者们执行用例的数据集。除此之外，MNIST数据集是一个相对较小的数据集，可以在你的笔记本CPUs上面直接执行





## 题目要求： 

- [ ] 请使用代码模板rbm.py，补全缺失部分，尽量不改动主体部分。
- [ ] 推荐使用python 及numpy 编写主要逻辑代码，适度使用框架。





## 项目简介

* 实现了 RBM 模型的基础结构，包括权重初始化、正负相传播、Contrastive Divergence（对比散度）训练算法、Gibbs 采样生成图像。
* 使用 `mnist_bin.npy` 作为输入数据，它是经过二值化处理的 MNIST 手写数字数据（60000 张 28×28 图片，像素值为 0 或 1）。

## 文件说明

| 文件名             | 说明                         |
| --------------- | -------------------------- |
| `rbm.py`        | 主程序，包含 `RBM` 类及训练与采样流程     |
| `mnist_bin.npy` | 二值化后的 MNIST 训练数据（NumPy 格式） |

## 🛠 依赖环境

* Python 2.7
* NumPy

>  注：本项目使用的是 Python 2.7，不兼容 Python 3，请确保环境正确。

安装 NumPy（如果尚未安装）：

```bash
pip install numpy
```

## 运行方式

确保当前目录中包含文件 `mnist_bin.npy`，然后运行：

```bash
python rbm.py
```

输出将包含：

* 数据维度打印信息（应为 `(60000, 28, 28)`）
* 模型训练过程（内部无显式打印）
* 最终输出为一张 28×28 的图像数据（NumPy 数组形式）

## 示例用途

虽然当前脚本未将采样图像保存为可视化文件，但你可以在训练完成后将 `rbm.sample()` 的输出使用 matplotlib 进行可视化：

```python
import matplotlib.pyplot as plt
plt.imshow(s, cmap='gray')
plt.title('Sampled Image from RBM')
plt.show()
```

## 模型参数说明

* `n_hidden=2`：隐藏层神经元数量，较小仅用于演示。
* `n_observe=784`：输入图像的像素数量（28×28）。
* 学习率：`0.1`
* 批大小：`100`
* 训练轮数：`10`

## 参考

* Hinton, G. E. (2002). "Training products of experts by minimizing contrastive divergence"
* MNIST 数据集：[http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
