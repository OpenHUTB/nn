#!/usr/bin/env python
# coding: utf-8

# # Tensorflow2.0 小练习

import tensorflow as tf
import numpy as np


# ## 实现softmax函数

def softmax(x):
    ##########
    '''实现softmax函数，只要求对最后一维归一化，
    不允许用tf自带的softmax函数'''
    ##########
    # 计算指数，为了避免数值溢出，减去最大值
    x_exp = tf.exp(x - tf.reduce_max(x, axis=-1, keepdims=True))
    # 计算归一化因子
    x_sum = tf.reduce_sum(x_exp, axis=-1, keepdims=True)
    # 计算softmax概率
    prob_x = x_exp / x_sum
    return prob_x

test_data = np.random.normal(size=[10, 5])
assert tf.reduce_all((softmax(test_data).numpy() - tf.nn.softmax(test_data, axis=-1).numpy())**2 < 0.0001).numpy()


# ## 实现sigmoid函数

def sigmoid(x):
    ##########
    '''实现sigmoid函数， 不允许用tf自带的sigmoid函数'''
    ##########
    prob_x = 1.0 / (1.0 + tf.exp(-x))
    return prob_x

test_data = np.random.normal(size=[10, 5])
assert tf.reduce_all((sigmoid(test_data).numpy() - tf.nn.sigmoid(test_data).numpy())**2 < 0.0001).numpy()


# ## 实现 softmax 交叉熵loss函数

def softmax_ce(x, label):
    ##########
    '''实现 softmax 交叉熵loss函数， 不允许用tf自带的softmax_cross_entropy函数'''
    ##########
    # 计算log概率（加一个小常数避免log(0)）
    log_prob = tf.math.log(x + 1e-10)
    # 计算每个样本的损失（选择对应标签的log概率）
    n_samples = tf.shape(label)[0]
    loss = -tf.reduce_mean(tf.reduce_sum(label * log_prob, axis=-1))
    return loss

test_data = np.random.normal(size=[10, 5])
prob = tf.nn.softmax(test_data)
label = np.zeros_like(test_data)
label[np.arange(10), np.random.randint(0, 5, size=10)] = 1.

assert (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=test_data, labels=label)) - 
        softmax_ce(prob, label))**2 < 0.0001


# ## 实现 sigmoid 交叉熵loss函数

def sigmoid_ce(x, label):
    ##########
    '''实现 sigmoid 交叉熵loss函数， 不允许用tf自带的sigmoid_cross_entropy函数'''
    ##########
    # 计算二元交叉熵损失
    loss = -tf.reduce_mean(label * tf.math.log(x + 1e-10) + 
                          (1 - label) * tf.math.log(1 - x + 1e-10))
    return loss

test_data = np.random.normal(size=[10])
prob = tf.nn.sigmoid(test_data)
label = np.random.randint(0, 2, 10).astype(test_data.dtype)
print(label)

assert (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=test_data, labels=label)) - 
        sigmoid_ce(prob, label))**2 < 0.0001
