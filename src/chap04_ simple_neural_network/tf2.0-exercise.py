#!/usr/bin/env python
# coding: utf-8

# # Tensorflow2.0 小练习

# In[2]:


import tensorflow as tf
import numpy as np


# ## 实现softmax函数

# In[6]:


def softmax(x):
    ##########
    '''实现softmax函数，只要求对最后一维归一化，
    不允许用tf自带的softmax函数'''
    x_max = tf.reduce_max(x, axis=-1, keepdims=True)
    e_x = tf.exp(x - x_max)
    prob_x = e_x / tf.reduce_sum(e_x, axis=-1, keepdims=True)
    ##########
    return prob_x

test_data = np.random.normal(size=[10, 5])
(softmax(test_data).numpy() - tf.nn.softmax(test_data, axis=-1).numpy())**2 <0.0001


# ## 实现sigmoid函数

# In[9]:


def sigmoid(x):
    ##########
    '''实现sigmoid函数， 不允许用tf自带的sigmoid函数'''
    prob_x = 1 / (1 + tf.exp(-x))
    ##########
    return prob_x

test_data = np.random.normal(size=[10, 5])
(sigmoid(test_data).numpy() - tf.nn.sigmoid(test_data).numpy())**2 < 0.0001


# ## 实现 softmax 交叉熵loss函数

# In[32]:


def softmax_ce(x, label):
    ##########
    '''实现 softmax 交叉熵loss函数， 不允许用tf自带的softmax_cross_entropy函数'''
    ##########
    # 实现 softmax 交叉熵loss函数， 不允许用tf自带的softmax_cross_entropy函数
    # 定义一个极小值，用于避免对数运算时出现数值问题
    epsilon = 1e-8
    # 计算 softmax 交叉熵损失，先计算 label 与 log(x + epsilon) 的乘积，再对最后一维求和
    # 最后对所有样本求平均值并取负
    loss = -tf.reduce_mean(tf.reduce_sum(label * tf.math.log(x + epsilon), axis=1))
    ##########
    return loss

test_data = np.random.normal(size=[10, 5])
prob = tf.nn.softmax(test_data)
label = np.zeros_like(test_data)
label[np.arange(10), np.random.randint(0, 5, size=10)] = 1.0
# 比较自定义的 softmax 交叉熵损失函数和 TensorFlow 自带的 softmax 交叉熵损失函数的输出结果
# 计算两者差值的平方，并判断是否小于 0.0001，以此验证自定义函数的正确性
((tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(label, test_data))
  - softmax_ce(prob, label))**2 < 0.0001).numpy()

# ## 实现 sigmoid 交叉熵loss函数
# In[46]:

def sigmoid_ce(x, label):
    ##########
    '''实现 softmax 交叉熵loss函数， 不允许用tf自带的softmax_cross_entropy函数'''
     # 计算 sigmoid 交叉熵损失，根据公式 - [y * log(p) + (1 - y) * log(1 - p)] 计算
    # 其中 y 是标签，p 是预测概率，最后对所有样本求平均值并取负
    epsilon = 1e-8
    loss = -tf.reduce_mean(
        label * tf.math.log(x + epsilon) + 
        (1 - label) * tf.math.log(1 - x + epsilon)
    )
    ##########
    return loss
# 生成一个形状为 [10] 的随机正态分布数据作为测试数据
test_data = np.random.normal(size=[10])
# 使用 TensorFlow 自带的 sigmoid 函数将测试数据转换为概率
prob = tf.nn.sigmoid(test_data)
# 生成随机的 0 或 1 标签，并将数据类型转换为与测试数据相同
label = np.random.randint(0, 2, 10).astype(test_data.dtype)
print (label)
# 比较自定义的 sigmoid 交叉熵损失函数和 TensorFlow 自带的 sigmoid 交叉熵损失函数的输出结果
# 计算两者差值的平方，并判断是否小于 0.0001，以此验证自定义函数的正确性
((tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(label, test_data))- sigmoid_ce(prob, label))**2 < 0.0001).numpy()


# In[ ]:




