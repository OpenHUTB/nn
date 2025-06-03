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
        # 第一层：输入层(784) -> 隐藏层(256)
        self.W1 = tf.Variable(tf.random.normal([784, 256], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([256]))
        
        # 第二层：隐藏层(256) -> 隐藏层(128)
        self.W2 = tf.Variable(tf.random.normal([256, 128], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([128]))
        
        # 第三层：隐藏层(128) -> 输出层(10)
        self.W3 = tf.Variable(tf.random.normal([128, 10], stddev=0.1))
        self.b3 = tf.Variable(tf.zeros([10]))
        ####################
        
    def __call__(self, x):
        ####################
        '''实现模型函数体，返回未归一化的logits'''
        # 将输入展平
        x = tf.reshape(x, [-1, 784])
        
        # 第一层：线性变换 + ReLU激活
        h1 = tf.matmul(x, self.W1) + self.b1
        h1 = tf.nn.relu(h1)
        
        # 第二层：线性变换 + ReLU激活
        h2 = tf.matmul(h1, self.W2) + self.b2
        h2 = tf.nn.relu(h2)
        
        # 第三层：线性变换（输出层）
        logits = tf.matmul(h2, self.W3) + self.b3
        ####################
        return logits
        
model = myModel()

# 使用Adam优化器，设置合适的学习率
optimizer = optimizers.Adam(learning_rate=0.001)


# ## 计算 loss

# In[13]:


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
    trainable_vars = [model.W1, model.W2, model.W3, model.b1, model.b2, model.b3]
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

