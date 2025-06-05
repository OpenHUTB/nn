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
        # 定义第一层的权重矩阵
        self.W1 = tf.Variable(tf.random.normal([784, 128], stddev=0.1))
        # 定义第一层的偏置向量
        self.b1 = tf.Variable(tf.zeros([128]))
        # 定义第二层的权重矩阵
        self.W2 = tf.Variable(tf.random.normal([128, 10], stddev=0.1))
        # 定义第二层的偏置向量
        self.b2 = tf.Variable(tf.zeros([10]))
        
    def __call__(self, x):
        '''实现模型函数体，返回未归一化的logits，这里未实现具体运算逻辑，需补充'''
        #logits = None
        #return logits
        # 展平为[batch_size, 784]
        x = tf.reshape(x, [-1, 784])
        # 隐藏层+ReLU
        h = tf.nn.relu(x @ self.W1 + self.b1) 
        # 输出层（未归一化）
        logits = h @ self.W2 + self.b2         
        return logits
        
model = myModel()

# 使用Adam优化器，用于训练过程中更新模型参数
optimizer = optimizers.Adam()

# ## 计算 loss

# In[13]:
# 使用tf.function装饰器将函数编译为TensorFlow图，提高执行效率
@tf.function
def compute_loss(logits, labels):
    # 计算稀疏softmax交叉熵损失，并求平均值
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = logits, labels = labels))

# 使用tf.function装饰器将函数编译为TensorFlow图，提高执行效率
@tf.function
def compute_accuracy(logits, labels):
    # 对logits在axis=1维度上取最大值的索引，得到预测结果
    predictions = tf.argmax(logits, axis=1)
    # 计算预测结果与真实标签相等的比例，得到准确率
    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))

# 使用tf.function装饰器将函数编译为TensorFlow图，提高执行效率
@tf.function
def train_one_step(model, optimizer, x, y):
    # 使用GradientTape记录计算图，用于自动求导
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(logits, y)

    # 计算可训练变量的梯度，这里的变量声明在模型中未完善，当前为假设的变量
    trainable_vars = [model.W1, model.W2, model.b1, model.b2]
    grads = tape.gradient(loss, trainable_vars)
    # 使用梯度更新可训练变量
    for g, v in zip(grads, trainable_vars):
        v.assign_sub(0.01*g)

    accuracy = compute_accuracy(logits, y)
    # loss and accuracy is scalar tensor
    return loss, accuracy

# 使用tf.function装饰器将函数编译为TensorFlow图，提高执行效率
@tf.function
def test(model, x, y):
    logits = model(x) # 计算预测结果与真实标签之间的损失值
    loss = compute_loss(logits, y) # compute_loss函数应实现具体的损失计算逻辑
    accuracy = compute_accuracy(logits, y) # 计算预测结果的准确率，compute_accuracy函数应实现准确率的计算逻辑
    return compute_loss(logits, y), compute_accuracy(logits, y)


# ## 实际训练

# In[14]:
# 加载MNIST数据集
train_data, test_data = mnist_dataset()
# 进行50个epoch的训练
for epoch in range(50):
    # 执行一次训练步骤，传入模型、优化器、训练数据及标签
    loss, accuracy = train_one_step(model, optimizer, 
                                    tf.constant(train_data[0], dtype=tf.float32),  # 图像数据
                                    tf.constant(train_data[1], dtype=tf.int64)) # 图像数据
    print('epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())
# 在测试集上测试模型
loss, accuracy = test(model, 
                      tf.constant(test_data[0], dtype=tf.float32), 
                      tf.constant(test_data[1], dtype=tf.int64))

print('test loss', loss.numpy(), '; accuracy', accuracy.numpy())
