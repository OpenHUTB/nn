#!/usr/bin/env python
# coding: utf-8

# ## 准备数据

# In[29]:
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# 设置TensorFlow日志级别，只显示错误信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def mnist_dataset():
    """
    加载并预处理MNIST数据集，返回训练集和测试集的TensorFlow Dataset对象。

    返回:
        ds (tf.data.Dataset): 处理后的训练数据集。
        test_ds (tf.data.Dataset): 处理后的测试数据集。
    """
    (x, y), (x_test, y_test) = datasets.mnist.load_data()   # 加载MNIST手写数字数据集，包含60,000张训练图像和10,000张测试图像
    x = x.reshape(x.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.take(20000).shuffle(20000).batch(32)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(prepare_mnist_features_and_labels)
    test_ds = test_ds.take(20000).shuffle(20000).batch(20000)
    return ds, test_ds


def prepare_mnist_features_and_labels(x, y):
    """
    对MNIST数据集的特征和标签进行预处理。

    参数:
        x: 输入图像数据。
        y: 对应的标签。

    返回:
        x: 归一化后的图像数据。
        y: 转换为整型的标签。
    """
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y


# ## 建立模型

# In[24]:
class MyConvModel(keras.Model):
    """
    自定义卷积神经网络模型，用于图像分类任务。

    网络结构：
    Conv2D(32) → Conv2D(64) → MaxPooling2D → Flatten → Dense(100) → Dense(10)
    """

    def __init__(self):
        super(MyConvModel, self).__init__()
        self.l1_conv = Conv2D(32, (5, 5), activation='relu', padding='same')
        self.l2_conv = Conv2D(64, (5, 5), activation='relu', padding='same')
        self.pool = MaxPooling2D(pool_size=(2, 2), strides=2)
        self.flat = Flatten()
        self.dense1 = layers.Dense(100, activation='tanh')
        self.dense2 = layers.Dense(10)

    @tf.function
    def call(self, x):
        """
        模型的前向传播过程。

        参数:
            x: 输入数据。

        返回:
            logits: 模型输出的logits。
        """
        h1 = self.l1_conv(x)
        h1_pool = self.pool(h1)
        h2 = self.l2_conv(h1_pool)
        h2_pool = self.pool(h2)
        flat_h = self.flat(h2_pool)
        dense1 = self.dense1(flat_h)
        logits = self.dense2(dense1)
        return logits


model = MyConvModel()
optimizer = optimizers.Adam()


# ## 定义loss以及train loop

# In[25]:
@tf.function
def compute_loss(logits, labels):
    """
    计算稀疏softmax交叉熵损失。

    参数:
        logits: 模型输出的logits。
        labels: 真实标签。

    返回:
        loss: 计算得到的损失值。
    """
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
    )


@tf.function
def compute_accuracy(logits, labels):
    """
    计算模型的准确率。

    参数:
        logits: 模型输出的logits。
        labels: 真实标签。

    返回:
        accuracy: 模型的准确率。
    """
    predictions = tf.argmax(logits, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))


@tf.function
def train_one_step(model, optimizer, x, y):
    """
    执行单步训练过程。

    参数:
        model: 模型实例。
        optimizer: 优化器实例。
        x: 输入数据。
        y: 真实标签。

    返回:
        loss: 训练损失。
        accuracy: 训练准确率。
    """
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(logits, y)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    accuracy = compute_accuracy(logits, y)
    return loss, accuracy


@tf.function
def test_step(model, x, y):
    """
    执行单步测试过程。

    参数:
        model: 模型实例。
        x: 输入数据。
        y: 真实标签。

    返回:
        loss: 测试损失。
        accuracy: 测试准确率。
    """
    logits = model(x)
    loss = compute_loss(logits, y)
    accuracy = compute_accuracy(logits, y)
    return loss, accuracy


def train(epoch, model, optimizer, ds):
    """
    训练模型一个完整的epoch（遍历整个训练数据集一次）。
    参数:
        epoch: 当前训练轮数（整数），用于打印进度信息
        model: 要训练的模型实例（继承自tf.keras.Model）
        optimizer: 优化器实例（如Adam/SGD等）
        ds: 训练数据集（通常为tf.data.Dataset对象）
    返回:
        loss: 本epoch最后一个batch的损失值（tf.Tensor）
        accuracy: 本epoch最后一个batch的准确率（tf.Tensor）
    """
    # 初始化累计变量（实际仅记录最后一个batch的值）
    loss = 0.0      # 损失值初始化为0（浮点数）
    accuracy = 0.0   # 准确率初始化为0（浮点数）
    
    # 遍历数据集中的所有batch
    for step, (x, y) in enumerate(ds):
        # 执行单步训练，返回当前batch的loss和accuracy
        loss, accuracy = train_one_step(model, optimizer, x, y)
        
        # 每500个batch打印一次训练进度
        if step % 500 == 0:  # 使用取模运算控制打印频率
            # 将Tensor转换为numpy值打印
            print('epoch', epoch, ': loss', loss.numpy(),
                  '; accuracy', accuracy.numpy())
    
    # 返回最后一个batch的loss和accuracy（注意不是epoch平均值）
    return loss, accuracy


def test(model, ds):
    """
    测试模型。

    参数:
        model: 模型实例。
        ds: 测试数据集。

    返回:
        loss: 测试损失。
        accuracy: 测试准确率。
    """
    loss = 0.0
    accuracy = 0.0
    
    for step, (x, y) in enumerate(ds):
        loss, accuracy = test_step(model, x, y)

    print('test loss', loss.numpy(), '; accuracy', accuracy.numpy())
    return loss, accuracy


# # 训练

# In[26]:、
#调用mnist_dataset()函数获取处理好的MNIST训练集和测试集，train_ds是训练数据集，test_ds是测试数据集
train_ds, test_ds = mnist_dataset()
for epoch in range(2):
    loss, accuracy = train(epoch, model, optimizer, train_ds)
loss, accuracy = test(model, test_ds)
