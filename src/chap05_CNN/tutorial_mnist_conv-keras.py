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
        probs: 输出概率分布张量，形状为 [batch_size, num_classes]，
               每个样本的各类别概率之和为1
    """
    # 第一层卷积操作：使用3x3卷积核提取局部特征
    # 输入形状：[N,H,W,C] -> 输出形状：[N,H,W,conv1_filters]
    h1 = self.l1_conv(x)  # l1_conv 是第一个卷积层实例
    
    # 第一层最大池化：2x2窗口下采样，减少空间维度
    # 输出形状变为输入的一半（如[N,128,128,32]->[N,64,64,32]）
    h1_pool = self.pool(h1)  # pool 是 MaxPooling2D 层实例
    
    # 第二层卷积操作：在更高层次提取特征
    # 使用更多卷积核捕获更复杂的模式
    h2 = self.l2_conv(h1_pool)  # l2_conv 是第二个卷积层实例
    
    # 第二层最大池化：进一步压缩空间信息
    h2_pool = self.pool(h2)
    
    # 展平操作：将多维特征图转换为一维特征向量
    # 例如 [N,7,7,64] -> [N,3136]
    flat_h = self.flat(h2_pool)  # flat 是 Flatten 层实例
    
    # 第一个全连接层：进行高级特征组合和非线性变换
    # 通常使用ReLU等激活函数引入非线性
    dense1 = self.dense1(flat_h)  # dense1 是第一个Dense层
    
    # 输出层：生成各分类的原始得分（logits）
    # 不使用激活函数，保持线性输出
    logits = self.dense2(dense1)  # dense2 是输出层
    
    # 应用softmax将logits转换为概率分布
    # axis=-1 表示在最后一个维度（类别维度）进行归一化
    probs = tf.nn.softmax(logits, axis=-1)
    
    return probs

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
