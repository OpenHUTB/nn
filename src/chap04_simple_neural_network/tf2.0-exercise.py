#!/usr/bin/env python
# coding: utf-8

# # Tensorflow2.0 小练习

import numpy as np
import tensorflow as tf

# ## 实现softmax函数

def softmax(x: tf.Tensor) -> tf.Tensor:
    """
    实现数值稳定的 softmax 函数，仅在最后一维进行归一化。
    
    参数:
        x: 输入张量，任意形状，通常最后一维表示分类 logits。
    
    返回:
        与输入形状相同的 softmax 概率分布张量。
    """
    x = tf.cast(x, tf.float32)

    # 数值稳定性处理：减去最大值避免指数爆炸
    max_per_row = tf.reduce_max(x, axis=-1, keepdims=True)
    shifted_logits = x - max_per_row

    # 计算 softmax
    exp_logits = tf.exp(shifted_logits)
    sum_exp = tf.reduce_sum(exp_logits, axis=-1, keepdims=True)
    softmax_output = exp_logits / sum_exp

    return softmax_output

# 生成测试数据，形状为 [10, 5] 的正态分布随机数
test_data = np.random.normal(size=[10, 5])
# 比较自定义的softmax函数结果和tf自带的结果，误差小于 0.0001 则认为相等
(softmax(test_data).numpy() - tf.nn.softmax(test_data, axis=-1).numpy())**2 < 0.0001

# ## 实现sigmoid函数

def sigmoid(x):
    ##########
    '''实现sigmoid函数， 不允许用tf自带的sigmoid函数'''
    x = tf.cast(x, tf.float32) #将输入x转换为float32类型，确保数值计算的精度和类型一致性。
    prob_x = 1 / (1 + tf.exp(-x))           # sigmoid 数学定义：1 / (1 + e^{-x})
    ##########
    return prob_x

# 生成测试数据，形状为 [10, 5] 的正态分布随机数
test_data = np.random.normal(size=[10, 5])
# 比较自定义的sigmoid函数结果和tf自带的结果，误差小于 0.0001 则认为相等
(sigmoid(test_data).numpy() - tf.nn.sigmoid(test_data).numpy())**2 < 0.0001

# ## 实现 softmax 交叉熵loss函数

def softmax_ce(x, label):
    ##########
    '''实现 softmax 交叉熵loss函数， 不允许用tf自带的softmax_cross_entropy函数'''
    #参数logits: 未经Softmax的原始输出（logits）
    #参数label: one-hot格式的标签
    epsilon = 1e-8
    logits = tf.cast(logits, tf.float32)
    label = tf.cast(label, tf.float32)
    # 数值稳定处理：减去最大值
    logits_max = tf.reduce_max(logits, axis=-1, keepdims=True)
    stable_logits = logits - logits_max
    # 计算Softmax概率
    exp_logits = tf.exp(stable_logits)
    prob = exp_logits / tf.reduce_sum(exp_logits, axis=-1, keepdims=True)
    # 计算交叉熵
    loss = -tf.reduce_mean(tf.reduce_sum(label * tf.math.log(prob + epsilon), axis=1))
    ##########
    return loss

# 生成测试数据，形状为 [10, 5] 的正态随机数
test_data = np.random.normal(size=[10, 5])
# 进行softmax转换
prob = tf.nn.softmax(test_data)
# 生成标签，每个样本只有一个类别为 1
label = np.zeros_like(test_data)
label[np.arange(10), np.random.randint(0, 5, size=10)] = 1.0
# 比较自定义的损失值和tf自带结果，误差小于 0.0001 则认为相等
((tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(label, test_data))
  - softmax_ce(prob, label))**2 < 0.0001).numpy()

# ## 实现 sigmoid 交叉熵loss函数

def sigmoid_ce(x, label):
    ##########
    '''实现 softmax 交叉熵loss函数， 不允许用tf自带的softmax_cross_entropy函数'''
    epsilon = 1e-8
    x = tf.cast(x, tf.float32)
    label = tf.cast(label, tf.float32) # 将标签转换为float32
    # 计算sigmoid交叉熵损失，先计算每个样本的交叉熵，再求平均值
    loss = -tf.reduce_mean(
        label * tf.math.log(x + epsilon) +
        (1 - label) * tf.math.log(1 - x + epsilon)
    )
    ##########
    return loss

# 生成测试数据，形状为 [10] 的正态随机数
test_data = np.random.normal(size=[10])
# 进行sigmoid转换
prob = tf.nn.sigmoid(test_data)
# 生成标签，取值为 0 或 1
label = np.random.randint(0, 2, 10).astype(test_data.dtype)
print(label)
# 比较自定义的损失值和tf自带的结果，误差小于 0.0001 则认为相等
((tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(label, test_data)) - sigmoid_ce(prob, label))**2 < 0.0001).numpy()




