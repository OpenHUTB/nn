#!/usr/bin/env python
# coding: utf-8

# # Tensorflow2.0 小练习

import tensorflow as tf
import numpy as np


# ## 实现softmax函数

def softmax(x):
    """
    实现数值稳定的softmax函数，对输入张量的最后一维进行归一化。
    不允许使用 TensorFlow 自带的 softmax 函数。
    
    参数:
        x (tf.Tensor): 输入张量
        
    返回:
        tf.Tensor: softmax处理后的概率分布
    """
    # 检查输入是否为张量
    if not tf.is_tensor(x):
        x = tf.convert_to_tensor(x)
    
    # 计算每个元素的指数值，减去最大值以提高数值稳定性
    # 使用keepdims=True确保维度正确
    x_max = tf.reduce_max(x, axis=-1, keepdims=True)
    exp_x = tf.exp(x - x_max)
    
    # 计算softmax值，添加小的epsilon值避免除零错误
    # 1e-10是科学计数法，表示1乘以10的负10次方
    sum_exp = tf.reduce_sum(exp_x, axis=-1, keepdims=True)
    return exp_x / (sum_exp + 1e-10)


# 测试 softmax 实现是否正确，使用随机数据对比 TensorFlow 的实现
test_data = np.random.normal(size=[10, 5])
(softmax(test_data).numpy() - tf.nn.softmax(test_data, axis=-1).numpy())**2 < 0.0001


# ## 实现sigmoid函数

def sigmoid(x):
    exp_neg_x = tf.exp(-x)# 计算 -x 的指数
    prob_x = 1.0 / (1.0 + exp_neg_x) # 计算 sigmoid 函数值
    return prob_x


# 测试 sigmoid 实现是否正确
test_data = np.random.normal(size=[10, 5])
(sigmoid(test_data).numpy() - tf.nn.sigmoid(test_data).numpy())**2 < 0.0001


# ## 实现 softmax 交叉熵loss函数

def softmax_ce(x, label):
    ##########
    '''实现 softmax 交叉熵loss函数， 不允许用tf自带的softmax_cross_entropy函数'''
    ##########
    # 使用 clip 避免 log(0) 产生数值不稳定
    x = tf.clip_by_value(x, 1e-10, 1.0)
    # 计算交叉熵损失：-sum(y_true * log(y_pred))
    loss = -tf.reduce_mean(tf.reduce_sum(label * tf.math.log(x), axis=-1))
    ##########
    return loss


# 构造测试数据并验证 softmax_ce 函数正确性
test_data = np.random.normal(size=[10, 5])
# 得到 softmax 概率
prob = tf.nn.softmax(test_data)
# 创建 one-hot 标签
label = np.zeros_like(test_data)
# 每行随机一个位置设为 1
label[np.arange(10), np.random.randint(0, 5, size=10)] = 1.0  

# 对比手动实现和 TensorFlow 实现的 softmax 交叉熵结果
((tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(label, test_data))
  - softmax_ce(prob, label))**2 < 0.0001).numpy()


# ## 实现 sigmoid 交叉熵loss函数
def sigmoid_ce(x, label):
    ##########
    '''实现 softmax 交叉熵loss函数， 不允许用tf自带的softmax_cross_entropy函数'''
    ##########
    # clip 避免 log(0) 的数值不稳定问题
    x = tf.clip_by_value(x, 1e-10, 1.0 - 1e-10)
    # 计算二分类交叉熵损失
    loss = -tf.reduce_mean(
        label * tf.math.log(x) + (1 - label) * tf.math.log(1 - x)
    )
    ##########
    return loss


# 构造测试数据并验证 sigmoid_ce 函数正确性
test_data = np.random.normal(size=[10])
# 得到 sigmoid 概率
prob = tf.nn.sigmoid(test_data)  
# 随机生成 0 或 1 的标签
label = np.random.randint(0, 2, 10).astype(test_data.dtype)   
print(label)

# 对比手动实现和 TensorFlow 实现的 sigmoid 交叉熵结果
((tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(label, test_data))
  - sigmoid_ce(prob, label))**2 < 0.0001).numpy()
