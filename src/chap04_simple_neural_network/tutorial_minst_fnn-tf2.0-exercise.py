#!/usr/bin/env python
# coding: utf-8

# ## 准备数据

# In[7]:
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

# 设置TensorFlow日志级别，减少无关信息输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

def mnist_dataset():
    # 加载MNIST数据集，包含训练集和测试集的图像及标签
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    
    # 对图像数据进行归一化处理，将像素值缩放到0到1之间
    x = x / 255.0
    x_test = x_test / 255.0
    
    return (x, y), (x_test, y_test)

# In[8]:
# 打印两个列表对应元素组成的元组列表，这里是示例代码，与后续模型训练测试无直接关系
print(list(zip([1, 2, 3, 4], ['a', 'b', 'c', 'd'])))


#定义了一个简单的两层神经网络模型，用于处理 MNIST 手写数字识别任务

# In[9]:
class myModel:
    def __init__(self):
        ####################
        '''声明模型对应的参数，这里未实现，实际应添加权重和偏置等参数声明'''
        # 定义第一层的权重矩阵 ; 权重矩阵 W1 的形状为 [784, 128]，初始值为均值为 0，标准差为 0.1 的正态分布随机数
        self.W1 = tf.Variable(tf.random.normal([784, 128], stddev=0.1))
        # 定义第一层的偏置向量 ; 偏置向量 b1 的形状为 [128]，初始化为全 0
        self.b1 = tf.Variable(tf.zeros([128]))
        # 定义第二层的权重矩阵
        self.W2 = tf.Variable(tf.random.normal([128, 10], stddev=0.1))
        # 定义第二层的偏置向量
        self.b2 = tf.Variable(tf.zeros([10]))
        
    def __call__(self, x):
        '''实现模型函数体，返回未归一化的 logits ，这里未实现具体运算逻辑，需补充
        参数：
            x : Tensor
            输入数据，形状为 [batch_size, 28, 28]（例如 28x28 像素的灰度图像）

        返回：
            logits : Tensor
            未经过归一化的输出 logits，形状为 [batch_size, 10]（对应 10 个分类的得分）
        '''
        # 展平为[batch_size, 784]
        x = tf.reshape(x, [-1, 784])
        
        # 隐藏层+ReLU,隐藏层计算：
        # 通过矩阵乘法（x @ self.W1）加上偏置项 self.b1，得到隐藏层的加权和
        # 使用 ReLU 激活函数增加非线性
        h = tf.nn.relu(x @ self.W1 + self.b1)

        #  输出层计算：全连接层（无激活函数，直接输出 logits）
        #  - 隐藏层特征（128维） × 权重矩阵 W2（128×10） → 分类评分（10维）
        #  - 未应用 softmax，便于后续使用 sparse_softmax_cross_entropy_with_logits 计算损失
        logits = h @ self.W2 + self.b2         
        return logits
        
model = myModel()

# 使用Adam优化器，用于训练过程中更新模型参数
optimizer = optimizers.Adam()

# ## 计算 loss

# In[13]:
# 使用tf.function装饰器将函数编译为TensorFlow图，提高执行效率
# @tf.function 将Python函数转换为TensorFlow计算图，加速计算并支持导出
@tf.function  
def compute_loss(logits, labels):
    """
    计算稀疏softmax交叉熵损失（分类任务常用损失函数）
    参数:
        logits: 模型最后一层的未归一化输出（未经过softmax），形状通常为[batch_size, num_classes]
        labels: 真实类别标签（非one-hot编码），形状为[batch_size]
    返回:
        标量损失值（已对batch求平均）
    """
    # 计算稀疏softmax交叉熵损失（不需要提前对labels做one-hot编码）
    # 内部实现等价于：先对logits做softmax，再计算交叉熵
    # 数学公式：loss = -log(exp(logits[true_class]) / Σexp(logits))
    # tf.reduce_mean() 对batch中所有样本的损失求平均
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,  # 模型原始输出（未归一化）
            labels=labels))  # 真实类别索引（0到num_classes-1）

# 使用tf.function装饰器将函数编译为TensorFlow图，提高执行效率
@tf.function
def compute_accuracy(logits, labels):
    # 对logits在axis=1维度上取最大值的索引，得到预测结果
    predictions = tf.argmax(logits, axis=1, output_type=tf.int64)  # 确保预测类型与标签一致
    # 计算预测结果与真实标签相等的比例，得到准确率
    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))

# 使用tf.function装饰器将函数编译为TensorFlow图，提高执行效率
@tf.function
def train_one_step(model, optimizer, x, y):
    """
    执行一次训练步骤，计算梯度并更新模型参数。
    """
    with tf.GradientTape() as tape:     # 记录计算图以计算梯度
        logits = model(x)               # 前向传播
        loss = compute_loss(logits, y)  # 计算损失

    grads = tape.gradient(loss, model.trainable_variables)            # 计算梯度
    optimizer.apply_gradients(zip(grads, model.trainable_variables))  # 更新参数

    accuracy = compute_accuracy(logits, y)   # 计算准确率
    return loss, accuracy

# 使用tf.function装饰器将函数编译为TensorFlow静态计算图，以提高执行效率
# 自动处理自动微分和GPU加速，特别适合模型测试阶段
@tf.function  
def test(model, x, y):
    """
    模型测试函数（前向传播+指标计算）
    
    Args:
        model: 训练好的模型实例
        x: 输入数据张量，shape=(batch_size, ...)
        y: 真实标签张量，shape=(batch_size, num_classes)或(batch_size,)
    
    Returns:
        tuple: (损失值, 准确率) 两个标量张量
    """
    # 模型前向传播计算预测logits（未归一化的预测值）
    logits = model(x)  # shape=(batch_size, num_classes)
    
    # 计算预测结果与真实标签之间的损失值（交叉熵等）
    # 注意：compute_loss函数需事先定义，接收(logits, labels)参数
    loss = compute_loss(logits, y)  
    
    # 计算预测准确率（比较预测类别和真实类别）
    # 注意：compute_accuracy函数需事先定义，接收(logits, labels)参数
    accuracy = compute_accuracy(logits, y)  
    
    # 显式返回计算结果（虽然loss/accuracy已计算，但避免重复计算）
    return loss, accuracy  # 返回类型: (tf.Tensor, tf.Tensor)


# ## 实际训练

# In[14]:
# 加载MNIST数据集
train_data, test_data = mnist_dataset()
# 进行50个epoch的训练
for epoch in range(50):
    # 执行一次训练步骤，传入模型、优化器、训练数据及标签
    loss, accuracy = train_one_step(model, optimizer, 
                                    tf.constant(train_data[0], dtype = tf.float32),  # 训练图像数据
                                    tf.constant(train_data[1], dtype = tf.int64))    # 训练标签数据
    print('epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())
# 在测试集上测试模型
loss, accuracy = test(model, 
                      tf.constant(test_data[0], dtype = tf.float32),  # 将测试特征数据转换为TensorFlow常量张量，数据类型为float32
                      tf.constant(test_data[1], dtype = tf.int64))    # 将测试标签数据转换为TensorFlow常量张量，数据类型为int64
# 打印测试集上的最终损失和准确率
print('test loss', loss.numpy(), '; accuracy', accuracy.numpy())
