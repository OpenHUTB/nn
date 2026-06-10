#!/usr/bin/env python
# coding: utf-8
# ## 准备数据

# In[1]:
# 导入操作系统相关功能模块，用于文件路径操作等系统级功能
import os
# 导入TensorFlow深度学习框架
import tensorflow as tf
# 从TensorFlow中导入Keras高级API（TensorFlow的内置Keras实现）
from tensorflow import keras

from tensorflow.keras import layers, optimizers, datasets # 导入Keras核心组件：层定义、优化器和常用数据集
# 从Keras中导入核心组件：
# layers - 包含各种神经网络层的实现
# optimizers - 包含各种优化算法（如SGD, Adam等）
# datasets - 包含常用数据集（如MNIST, CIFAR等）的便捷加载方式
from tensorflow.keras.layers import ( 
    Conv2D, Dense, Dropout, 
    Flatten, MaxPooling2D
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'# 设置环境变量，控制 TensorFlow 的日志输出等级
                                        # 'TF_CPP_MIN_LOG_LEVEL' 可以用来过滤不必要的日志信息，提升可读性
                                        # 值为 '2' 表示屏蔽 Info 和 Warning 级别的日志


def mnist_dataset():
    """
    加载并预处理MNIST数据集，将20000个样本打乱并分批次。
    Returns:
        ds (tf.data.Dataset): 处理后的训练数据集。
        test_ds (tf.data.Dataset): 处理后的测试数据集。
    """
    # 从 TensorFlow 的 datasets 模块中加载 MNIST 数据集
    # MNIST 是一个手写数字图片数据集，包含 60,000 个训练样本和 10,000 个测试样本
    # (x, y) 是训练集的图像和标签
    # (x_test, y_test) 是测试集的图像和标签
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    # 将训练集图像 x 调整形状（reshape）
    # 原始图像形状是 (num_samples, 28, 28)，即每张图是 28x28 的灰度图
    # 调整后变为 (num_samples, 28, 28, 1)，增加一个通道维度（channel），以适配卷积神经网络的输入要求
    x = x.reshape(x.shape[0], 28, 28, 1)
    # 对测试集图像 x_test 做同样的 reshape 操作
    # 同样变成 (num_samples, 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # 从训练数据 x 和 y 创建一个 TensorFlow Dataset 对象
    ds = tf.data.Dataset.from_tensor_slices((x, y))

    # 使用 map 函数对每个样本应用预处理函数 prepare_mnist_features_and_labels（通常包括归一化、类型转换等）
    ds = ds.map(prepare_mnist_features_and_labels)

    # 取出前 20000 个样本，并对其进行打乱（shuffle），然后按每批 100 个样本进行分批
    ds = ds.take(20000).shuffle(20000).batch(100)


    # 从测试数据 x_test 和 y_test 创建测试集 Dataset
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # 同样应用预处理函数对测试集进行处理
    test_ds = test_ds.map(prepare_mnist_features_and_labels)

    # 从测试集中取出 20000 个样本，进行打乱后一次性全部放入一个 batch（用于评估时批量推理）
    test_ds = test_ds.take(20000).shuffle(20000).batch(20000)  
    # 对取出的 20000 个样本进行随机打乱，shuffle 的参数 20000 表示缓冲区大小，用于随机打乱数据


    # 返回训练集和测试集
    return ds, test_ds


def prepare_mnist_features_and_labels(x, y):
    """
    准备MNIST数据特征和标签。

    Args:
        x: 输入图像数据。
        y: 对应的标签。

    Returns:
        x: 归一化后的图像数据。
        y: 转换为整型的标签。
    """
    x = tf.cast(x, tf.float32) / 255.0# 显式转换为float32后除255
    # 将标签转换为int64类型
    # 确保标签类型与损失函数要求匹配（如sparse_categorical_crossentropy需要int类型标签）
    y = tf.cast(y, tf.int64)# 使用int64避免平台相关差异
    return x, y


# ## 建立模型

# In[5]:
class MyConvModel(keras.Model):
    """
    自定义卷积神经网络模型。
    """

    def __init__(self):
        super(MyConvModel, self).__init__()
        self.l1_conv = Conv2D(32, (5, 5), activation='relu', padding='same')  # 第一层卷积层
        self.l2_conv = Conv2D(64, (5, 5), activation='relu', padding='same')  # 第二层卷积层
        self.pool = MaxPooling2D(pool_size=(2, 2), strides=2)                 # 最大池化层
        self.flat = Flatten()                                                 # 展平层
        self.dense1 = layers.Dense(100, activation='tanh')                    # 第一层全连接层
        self.dense2 = layers.Dense(10)                                        # 第二层全连接层

    @tf.function
    def call(self, x):
        """
        前向传播函数（定义模型的数据流向）。
        Args:
            x: 输入数据张量，形状通常为 [batch_size, height, width, channels]
        Returns:
            logits: 输出张量，形状为 [batch_size, num_classes]
        """
        h1 = self.l1_conv(x)
        h1_pool = self.pool(h1)
        h2 = self.l2_conv(h1_pool)
        h2_pool = self.pool(h2)
        flat_h = self.flat(h2_pool)
        dense1 = self.dense1(flat_h)
        logits = self.dense2(dense1)
        return logits
# 创建一个神经网络模型的实例
model = MyConvModel()
optimizer = optimizers.Adam()# 配置Adam优化器：自适应矩估计优化算法


# ## 编译， fit以及evaluate

# In[6]:
#这段代码片段展示了如何使用Keras API来编译、训练和评估一个神经网络模型。
# 编译模型，指定优化器、损失函数和评估指标
model.compile(
    # 指定优化器，例如 Adam 或 SGD
    optimizer=optimizer,
    # 指定损失函数，适用于多分类任务
    loss='sparse_categorical_crossentropy',
    # 指定评估指标，这里使用准确率
    metrics=['accuracy']
)
# 加载MNIST数据集，返回训练集和测试集
train_ds, test_ds = mnist_dataset()
# 训练模型，指定训练数据和训练轮数
model.fit(train_ds, epochs=5)
# 评估模型，指定测试数据
model.evaluate(test_ds)
