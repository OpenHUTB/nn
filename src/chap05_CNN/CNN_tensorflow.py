#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np

# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 数据预处理
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

learning_rate = 1e-4
keep_prob_rate = 0.7
max_epoch = 2000

class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层1
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=7,
            strides=1,
            padding='same',
            activation='relu'
        )
        self.pool1 = tf.keras.layers.MaxPool2D(
            pool_size=2,
            strides=2,
            padding='same'
        )
        
        # 卷积层2
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=5,
            strides=1,
            padding='same',
            activation='relu'
        )
        self.pool2 = tf.keras.layers.MaxPool2D(
            pool_size=2,
            strides=2,
            padding='same'
        )
        
        # 全连接层
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(1024, activation='relu')
        self.dropout = tf.keras.layers.Dropout(keep_prob_rate)
        self.fc2 = tf.keras.layers.Dense(10, activation='softmax')
    
    def call(self, x, training=False):
        x = tf.reshape(x, [-1, 28, 28, 1])
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        return x

# 创建模型和优化器
model = CNN()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# 定义损失函数
loss_fn = tf.keras.losses.CategoricalCrossentropy()

# 定义训练步骤
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, predictions

# 定义测试步骤
@tf.function
def test_step(x, y):
    predictions = model(x, training=False)
    loss = loss_fn(y, predictions)
    return loss, predictions

# 训练模型
for epoch in range(max_epoch):
    # 随机选择100个样本进行训练
    indices = np.random.randint(0, len(x_train), 100)
    batch_xs = x_train[indices]
    batch_ys = y_train[indices]
    
    # 训练步骤
    loss, predictions = train_step(batch_xs, batch_ys)
    
    # 每100步打印一次测试准确率
    if epoch % 100 == 0:
        test_loss, test_predictions = test_step(x_test[:1000], y_test[:1000])
        test_accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(test_predictions, 1), tf.argmax(y_test[:1000], 1)), tf.float32)
        )
        print(f'Epoch {epoch}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

