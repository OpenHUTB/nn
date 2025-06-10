#!/usr/bin/env python
# coding: utf-8
# ## 准备数据

# In[7]:

# 导入操作系统接口模块，用于文件/目录操作
import os

# 导入NumPy库，用于科学计算（尤其是多维数组操作）
import numpy as np

# 导入TensorFlow深度学习框架
import tensorflow as tf

# 从TensorFlow中导入Keras高级API
from tensorflow import keras

# 从Keras中导入核心组件：
# layers - 神经网络层（如全连接层、卷积层等）
# optimizers - 优化器（如SGD、Adam等）
# datasets - 内置数据集（如MNIST、CIFAR等）
from tensorflow.keras import layers, optimizers, datasets

# 设置TensorFlow日志级别，避免输出过多无关信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 可选值：{'0', '1', '2'}

def mnist_dataset():
    """
    加载MNIST数据集并进行预处理：
    - 划分为训练集和测试集
    - 像素值归一化到[0, 1]区间
    - 保持原始数据类型（图像为float32，标签为int64）
    """
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    # 归一化像素值到[0, 1]
    x = x / 255.0
    x_test = x_test / 255.0
    return (x, y), (x_test, y_test)

# In[8]:

# print(list(zip([1, 2, 3, 4], ['a', 'b', 'c', 'd'])))  # 测试代码，可忽略

# ## 建立模型

# In[9]:

class MyModel:
    def __init__(self):
        ####################
        '''声明模型对应的参数'''
        ####################
        # 初始化权重和偏置
        # 输入层784 -> 隐藏层128
        self.W1 = tf.Variable(tf.random.normal([784, 128], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([128]))
        # 隐藏层128 -> 输出层10
        self.W2 = tf.Variable(tf.random.normal([128, 10], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([10]))

    def __call__(self, x):
        ####################
        '''实现模型函数体，返回未归一化的logits'''
        ####################
        x = tf.reshape(x, [-1, 784])  # 将输入图像展平成一维向量
        h1 = tf.nn.relu(tf.matmul(x, self.W1) + self.b1)  # 第一层全连接+ReLU激活
        logits = tf.matmul(h1, self.W2) + self.b2         # 输出层全连接，未归一化logits
        return logits

# 实例化模型和优化器
model = MyModel()
optimizer = optimizers.Adam()  # 配置Adam优化器，自适应调整学习率，参数为默认值

# ## 计算 loss

# In[13]:

@tf.function
def compute_loss(logits, labels):
    """
    计算交叉熵损失（对所有样本的损失取平均，得到批次平均损失）
    logits: 模型输出的未归一化分数
    labels: 真实标签
    """
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels
        )
    )
# 计算稀疏标签的softmax交叉熵损失

@tf.function
def compute_accuracy(logits, labels):
    """
    计算准确率
    logits: 模型输出的未归一化分数
    labels: 真实标签
    """
    predictions = tf.argmax(logits, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))

@tf.function
def train_one_step(model, optimizer, x, y):
    """
    执行一次训练步骤，计算梯度并更新模型参数。

    参数:
        model: 模型实例
        optimizer: 优化器实例
        x: 输入数据
        y: 标签数据

    返回:
        loss: 训练损失
        accuracy: 训练准确率
    """
    with tf.GradientTape() as tape:
        logits = model(x)  # 前向传播，获取模型输出
        loss = compute_loss(logits, y)  # 计算损失

    # 计算梯度
    trainable_vars = [model.W1, model.W2, model.b1, model.b2]
    grads = tape.gradient(loss, trainable_vars)

    # 更新参数（使用固定学习率）
    for g, v in zip(grads, trainable_vars):
        v.assign_sub(0.01 * g)

    # 计算准确率
    accuracy = compute_accuracy(logits, y)

    # 返回损失和准确率（都是标量张量）
    return loss, accuracy

@tf.function
def test(model, x, y):
    """
    使用 TensorFlow 图模式执行模型评估的函数
    该函数计算模型在给定输入数据上的损失和准确率

    Args:
        model: 已构建的模型实例
        x: 输入特征张量，形状为 [batch_size, ...]
        y: 真实标签张量，形状为 [batch_size] 或 [batch_size, num_classes]

    Returns:
        loss: 标量张量，表示当前批次的损失值
        accuracy: 标量张量，表示当前批次的准确率
    """
    # 前向传播：通过模型计算预测输出（logits）
    logits = model(x)  # logits 是未经过 softmax 的原始输出

    # 计算损失值
    loss = compute_loss(logits, y)

    # 计算准确率
    accuracy = compute_accuracy(logits, y)

    return loss, accuracy

# ## 实际训练

# In[14]:

# 导入MNIST数据集，分割为训练集和测试集
train_data, test_data = mnist_dataset()

# 进行50个epoch的训练循环
for epoch in range(50):
    # 执行单步训练，传入模型、优化器、训练数据和标签
    # 将numpy数据转换为TensorFlow张量，指定数据类型为float32和int64
    loss, accuracy = train_one_step(
        model,
        optimizer,
        tf.constant(train_data[0], dtype=tf.float32),  # 训练图像数据
        tf.constant(train_data[1], dtype=tf.int64)     # 训练标签数据
    )
    print('epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())

# 在测试集上评估模型性能
loss, accuracy = test(
    model,
    tf.constant(test_data[0], dtype=tf.float32),# 将测试特征数据转换为TensorFlow常量张量，指定数据类型为float32
    tf.constant(test_data[1], dtype=tf.int64)
)
# .numpy() 将 TensorFlow 张量转换为 NumPy 数组（或 Python 标量）以便打印
print('test loss', loss.numpy(), '; accuracy', accuracy.numpy())
