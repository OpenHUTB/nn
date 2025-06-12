# !/usr/bin/env python
# coding: utf-8
# # Tensorflow2.0 小练习

# 导入 numpy 库，并简写为 np（标准约定）
import numpy as np
# 导入 TensorFlow 库，并简写为 tf（标准约定）
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
    x = tf.cast(x, tf.float32)  # 统一为float32类型，确保计算精度
    # 数值稳定性处理：减去最大值避免指数爆炸
    max_per_row = tf.reduce_max(x, axis=-1, keepdims=True)
    # 平移后的logits：每行最大值变为0，其他值为负数
    shifted_logits = x - max_per_row
    # 计算指数值
    exp_logits = tf.exp(shifted_logits)
    sum_exp = tf.reduce_sum(exp_logits, axis=-1, keepdims=True)
    return exp_logits / sum_exp

# 实现sigmoid函数
def sigmoid(x):
    """
    实现sigmoid函数， 不允许用tf自带的sigmoid函数
    """
    x = tf.cast(x, tf.float32)  # 确保数值计算的精度和类型一致性
    return 1 / (1 + tf.exp(-x))

# 实现 softmax 交叉熵loss函数
def softmax_ce(logits, label):
    """
    实现 softmax 交叉熵loss函数， 不允许用tf自带的softmax_cross_entropy函数
    参数logits: 未经Softmax的原始输出（logits）
    参数label: one-hot格式的标签
    """
    epsilon = 1e-8  # 用于数值稳定性，防止log(0)的情况
    logits = tf.cast(logits, tf.float32)
    label = tf.cast(label, tf.float32)
    # 数值稳定处理：减去最大值
    logits_max = tf.reduce_max(logits, axis=-1, keepdims=True)
    stable_logits = logits - logits_max
    # 计算Softmax概率
    exp_logits = tf.exp(stable_logits)
    prob = exp_logits / tf.reduce_sum(exp_logits, axis=-1, keepdims=True)
    # 计算交叉熵
    loss = -tf.reduce_mean(
        tf.reduce_sum(label * tf.math.log(prob + epsilon), axis=1)
    )
    return loss

# 实现 sigmoid 交叉熵loss函数
def sigmoid_ce(logits, labels):
    """
    实现 sigmoid 交叉熵 loss 函数（不使用 tf 内置函数）
    接收未经过 sigmoid 的 logits 输入
    """
    logits = tf.cast(logits, tf.float32)
    labels = tf.cast(labels, tf.float32)
    # 通过更稳定的方式实现 sigmoid 交叉熵
    loss = tf.reduce_mean(
        tf.nn.relu(logits) - logits * labels + 
        tf.math.log(1 + tf.exp(-tf.abs(logits)))
    )
    return loss

# 测试函数，用于比较自定义函数和tf自带函数的结果
def test_function(custom_func, tf_func, *args):
    custom_result = custom_func(*args)
    tf_result = tf_func(*args)
    error = ((tf_result - custom_result) ** 2 < 0.0001).numpy()
    return custom_result, tf_result, error

# 测试 softmax 函数
test_data_softmax = np.random.normal(size=[10, 5])
custom_softmax, tf_softmax, softmax_error = test_function(softmax, tf.nn.softmax, test_data_softmax, axis=-1)
print("Softmax 误差是否小于0.0001:", softmax_error)

# 测试 sigmoid 函数
test_data_sigmoid = np.random.normal(size=[10, 5])
custom_sigmoid, tf_sigmoid, sigmoid_error = test_function(sigmoid, tf.nn.sigmoid, test_data_sigmoid)
print("Sigmoid 误差是否小于0.0001:", sigmoid_error)

# 测试 softmax 交叉熵 loss 函数
test_logits = np.random.normal(size=[10, 5]).astype(np.float32)
label = np.zeros_like(test_logits, dtype=np.float32)
label[np.arange(10), np.random.randint(0, 5, size=10)] = 1.0
custom_softmax_ce, tf_softmax_ce, softmax_ce_error = test_function(softmax_ce, lambda x, y: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, x)), test_logits, label)
print("Softmax 交叉熵误差是否小于0.0001:", softmax_ce_error)

# 测试 sigmoid 交叉熵 loss 函数
test_data_sigmoid_ce = np.random.normal(size=[10]).astype(np.float32)
labels_sigmoid_ce = np.random.randint(0, 2, size=[10]).astype(np.float32)
custom_sigmoid_ce, tf_sigmoid_ce, sigmoid_ce_error = test_function(sigmoid_ce, lambda x, y: tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=x)), test_data_sigmoid_ce, labels_sigmoid_ce)
print("Sigmoid 交叉熵误差是否小于0.0001:", sigmoid_ce_error)