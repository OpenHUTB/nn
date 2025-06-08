#!/usr/bin/env python
# coding: utf-8
# ## 准备数据

# In[1]:
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def mnist_dataset():
    """
    加载并预处理MNIST数据集，将20000个样本打乱并分批次。
    Returns:
        ds (tf.data.Dataset): 处理后的训练数据集。
        test_ds (tf.data.Dataset): 处理后的测试数据集。
    """
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    x = x.reshape(x.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.take(20000).shuffle(20000).batch(100)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(prepare_mnist_features_and_labels)
    test_ds = test_ds.take(20000).shuffle(20000).batch(20000)
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
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
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
        self.pool = MaxPooling2D(pool_size=(2, 2), strides=2)                # 最大池化层
        self.flat = Flatten()                                                # 展平层
        self.dense1 = layers.Dense(100, activation='tanh')                   # 第一层全连接层
        self.dense2 = layers.Dense(10)                                       # 第二层全连接层

    @tf.function
    def call(self, x):
        """
        前向传播函数。

        Args:
            x: 输入数据。

        Returns:
            probs: 输出概率。
        """
        # 第一层卷积，提取特征
        h1 = self.l1_conv(x)
        # 第一层池化，降维
        h1_pool = self.pool(h1)
        # 第二层卷积，进一步提取特征
        h2 = self.l2_conv(h1_pool)
        # 第二层池化，再次降维
        h2_pool = self.pool(h2)
        # 展平特征图，准备输入全连接层
        # 将多维特征图展平为一维向量，以便输入全连接层。h2_pool 是前一层的输出
        flat_h = self.flat(h2_pool)
        # 第一个全连接层（Dense Layer），对展平后的特征进行非线性变换
        dense1 = self.dense1(flat_h)
        # 第二个全连接层（输出层），生成未归一化的分类得分（logits）
        logits = self.dense2(dense1)
        probs = tf.nn.softmax(logits, axis=-1)
        return probs

model = MyConvModel()
optimizer = optimizers.Adam()# 配置Adam优化器：自适应矩估计优化算法


# ## 编译， fit以及evaluate

# In[6]:
#这段代码片段展示了如何使用Keras API来编译、训练和评估一个神经网络模型。
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
train_ds, test_ds = mnist_dataset()
model.fit(train_ds, epochs=5)
model.evaluate(test_ds)
