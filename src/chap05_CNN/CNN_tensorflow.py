#!/usr/bin/env python
# coding: utf-8
# In[ ]:
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#使用input_data.read_data_sets函数加载MNIST数据集，'MNIST_data'是数据集存储的目录路径，one_hot=True表示将标签转换为one-hot编码格式
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

learning_rate = 1e-4 #学习率
keep_prob_rate = 0.7 # Dropout保留概率0.7
max_epoch = 2000 #最大训练轮数2000


def compute_accuracy(v_xs, v_ys):
    """
    计算模型在给定数据集上的准确率。

    参数:
        v_xs: 输入数据。
        v_ys: 真实标签。

    返回:
        result: 模型的准确率。
    """
    global prediction
    # 获取模型预测结果
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    # 比较预测与真实标签
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 运行准确率计算
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    """
    初始化权重变量。

    参数:
        shape: 权重的形状。

    返回:
        tf.Variable: 初始化后的权重变量。
    """
    # 使用截断正态分布初始化权重，stddev=0.1，有助于稳定训练
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)



def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, padding='SAME', strides=[1, 1, 1, 1]):
    """
    实现二维卷积操作，增加了参数灵活性和异常处理
    
    参数:
        x (tf.Tensor): 输入张量，形状为[batch, height, width, channels]
        W (tf.Tensor): 卷积核权重，形状为[filter_height, filter_width, in_channels, out_channels]
        padding (str): 填充方式，'SAME'或'VALID'
        strides (list): 步长列表，[1, stride_h, stride_w, 1]
        
    返回:
        tf.Tensor: 卷积结果
    """
    # 每一维度滑动步长全部是 1， padding 方式选择 same
    # 提示 使用函数  tf.nn.conv2d
    
    # 验证输入类型
    if not tf.is_tensor(x):
        x = tf.convert_to_tensor(x)
    
    # 验证padding参数
    if padding not in ['SAME', 'VALID']:
        raise ValueError(f"Invalid padding value: {padding}. Must be 'SAME' or 'VALID'.")
    
    # 执行卷积操作
    conv = tf.nn.conv2d(x, W, strides=strides, padding=padding)
    
    # 添加批归一化以提高训练稳定性
    # 注意：在实际应用中，是否使用批归一化取决于网络结构和需求
    # conv = tf.layers.batch_normalization(conv, training=is_training)
    
    return conv
    
def max_pool_2x2(x: tf.Tensor,
    pool_size: int = 2,
    strides: int = 2,
    padding: str = 'SAME',
    data_format: str = 'NHWC'
) -> tf.Tensor:
    # 验证参数合法性
    if padding not in ['SAME', 'VALID']:
        raise ValueError(f"padding must be 'SAME' or 'VALID', got {padding}.")
    if data_format not in ['NHWC', 'NCHW']:
        raise ValueError(f"data_format must be 'NHWC' or 'NCHW', got {data_format}.")
    
    # 构造池化核和步长参数
    if data_format == 'NHWC':
        ksize = [1, pool_size, pool_size, 1]
        strides = [1, strides, strides, 1]
    else:  # NCHW
        ksize = [1, 1, pool_size, pool_size]
        strides = [1, 1, strides, strides]
    
    return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding, data_format=data_format)

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) / 255.
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])

#  卷积层 1
## conv1 layer ##
W_conv1 = weight_variable([7, 7, 1, 32])                      # patch 7x7, in size 1, out size 32
b_conv1 = bias_variable([32])                     
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)                      # 卷积  自己选择 选择激活函数
h_pool1 = max_pool_2x2(h_conv1)                      # 池化               

# 卷积层 2
W_conv2 = weight_variable([5, 5, 32, 64])                       # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)                       # 卷积  自己选择 选择激活函数
h_pool2 = max_pool_2x2(h_conv2)                       # 池化

#  全连接层 1
# 定义全连接层1的权重（W_fc1），维度是 [7*7*64, 1024]：
# 输入是前一层池化输出展平后的长度（7x7x64），输出是1024个神经元
W_fc1 = weight_variable([7*7*64, 1024])

# 定义全连接层1的偏置（b_fc1），大小为1024，对应输出维度
b_fc1 = bias_variable([1024])

# 将上一层池化层的输出展平成一维向量，-1 表示自动计算 batch size
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

# 全连接计算 + ReLU激活函数
# matmul 矩阵乘法，得到的是一个 [batch_size, 1024] 的激活输出
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 应用 Dropout 防止过拟合，keep_prob 是保留节点的概率（在 feed_dict 中提供）
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 全连接层 2
## fc2 layer ##
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 交叉熵函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    
    for i in range(max_epoch):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob:keep_prob_rate})
        if i % 100 == 0:  #每 100 个迭代在测试集的前 1000 个样本上评估准确率
            print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))
