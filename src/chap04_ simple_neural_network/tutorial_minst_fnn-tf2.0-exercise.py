#!/usr/bin/env python
# coding: utf-8

# ## 准备数据

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

def mnist_dataset():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    # normalize
    x = x/255.0
    x_test = x_test/255.0
    
    return (x, y), (x_test, y_test)


print(list(zip([1, 2, 3, 4], ['a', 'b', 'c', 'd'])))


# ## 建立模型

class myModel:
    def __init__(self):
        ####################
        '''声明模型对应的参数'''
        ####################
        # 初始化权重和偏置
        self.W1 = tf.Variable(tf.random.normal(shape=(784, 256), stddev=0.1))
        self.b1 = tf.Variable(tf.zeros(shape=(256,)))
        self.W2 = tf.Variable(tf.random.normal(shape=(256, 10), stddev=0.1))
        self.b2 = tf.Variable(tf.zeros(shape=(10,)))
        
    def __call__(self, x):
        ####################
        '''实现模型函数体，返回未归一化的logits'''
        ####################
        # 展平输入图像 (28x28 -> 784)
        x = tf.reshape(x, shape=(-1, 784))
        # 第一层全连接
        h = tf.matmul(x, self.W1) + self.b1
        # ReLU激活函数
        h = tf.nn.relu(h)
        # 第二层全连接 (输出层)
        logits = tf.matmul(h, self.W2) + self.b2
        return logits
        
model = myModel()

optimizer = optimizers.Adam()


# ## 计算 loss

@tf.function
def compute_loss(logits, labels):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))

@tf.function
def compute_accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))

@tf.function
def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(logits, y)

    # compute gradient
    trainable_vars = [model.W1, model.W2, model.b1, model.b2]
    grads = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(grads, trainable_vars))

    accuracy = compute_accuracy(logits, y)

    # loss and accuracy is scalar tensor
    return loss, accuracy

@tf.function
def test(model, x, y):
    logits = model(x)
    loss = compute_loss(logits, y)
    accuracy = compute_accuracy(logits, y)
    return loss, accuracy


# ## 实际训练

train_data, test_data = mnist_dataset()
for epoch in range(50):
    # 批量训练
    batch_size = 64
    num_batches = len(train_data[0]) // batch_size
    total_loss = 0.0
    total_accuracy = 0.0
    
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        x_batch = train_data[0][start:end]
        y_batch = train_data[1][start:end]
        
        loss, accuracy = train_one_step(model, optimizer, 
                                        tf.constant(x_batch, dtype=tf.float32), 
                                        tf.constant(y_batch, dtype=tf.int64))
        total_loss += loss.numpy()
        total_accuracy += accuracy.numpy()
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
