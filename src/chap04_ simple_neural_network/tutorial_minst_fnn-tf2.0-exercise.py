#!/usr/bin/env python
# coding: utf-8
# ## 准备数据

# In[7]:

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


def mnist_dataset():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    #normalize
    
    x = x/255.0
    x_test = x_test/255.0
    
    return (x, y), (x_test, y_test)


# In[8]:

print(list(zip([1, 2, 3, 4], ['a', 'b', 'c', 'd'])))

# ## 建立模型

# In[9]:


class myModel:
    def __init__(self):
        ####################
        '''声明模型对应的参数'''
        ####################
    def __call__(self, x):
        ####################
        '''实现模型函数体，返回未归一化的logits'''
        ####################
        return logits
        
model = myModel()

optimizer = optimizers.Adam()


# ## 计算 loss

# In[13]:


@tf.function
def compute_loss(logits, labels):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = logits, labels = labels))

@tf.function
def compute_accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))


@tf.function
def train_one_step(model, optimizer, x, y):
    """
    执行一次训练步骤，计算梯度并更新模型参数。

    参数:
        model: 模型实例。
        optimizer: 优化器实例。
        x: 输入数据。
        y: 标签数据。

    返回:
        loss: 训练损失。
        accuracy: 训练准确率。
    """
    with tf.GradientTape() as tape:
        logits = model(x)  # 前向传播，获取模型输出
        loss = compute_loss(logits, y)  # 计算损失

    # 计算梯度
    trainable_vars = [model.W1, model.W2, model.b1, model.b2]
    grads = tape.gradient(loss, trainable_vars)

    # 更新参数
    for g, v in zip(grads, trainable_vars):
        v.assign_sub(0.01 * g)  # 使用固定学习率更新参数

    # 计算准确率
    accuracy = compute_accuracy(logits, y)

    # 返回损失和准确率（都是标量张量）
    return loss, accuracy

@tf.function
def test(model, x, y):
    logits = model(x)
    loss = compute_loss(logits, y)
    accuracy = compute_accuracy(logits, y)
    return loss, accuracy


# ## 实际训练

# In[14]:


train_data, test_data = mnist_dataset()
for epoch in range(50):
    loss, accuracy = train_one_step(model, optimizer, 
                                    tf.constant(train_data[0], dtype=tf.float32), 
                                    tf.constant(train_data[1], dtype=tf.int64))
    print('epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())
loss, accuracy = test(model, 
                      tf.constant(test_data[0], dtype=tf.float32), 
                      tf.constant(test_data[1], dtype=tf.int64))

print('test loss', loss.numpy(), '; accuracy', accuracy.numpy())

