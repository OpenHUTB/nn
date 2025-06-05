#!/usr/bin/env python
# coding: utf-8

# ## 准备数据

# In[1]:
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def mnist_dataset():
    """
    加载并预处理MNIST数据集，返回训练集和测试集的TensorFlow Dataset对象。

    Returns:
        ds (tf.data.Dataset): 处理后的训练数据集。
        test_ds (tf.data.Dataset): 处理后的测试数据集。
    """
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    x = x.reshape(x.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.take(20000).shuffle(20000).batch(100)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(prepare_mnist_features_and_labels)
    test_ds = test_ds.take(20000).shuffle(20000).batch(20000)
    return ds, test_ds


def prepare_mnist_features_and_labels(x, y):
    """
    对MNIST数据集的特征和标签进行预处理。

    Args:
        x: 输入图像数据。
        y: 对应的标签。

    Returns:
        x: 归一化后的图像数据。
        y: 转换为整型的标签。
    """
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y


# In[2]:
7 * 7 * 64


# ## 建立模型

# In[3]:
model = keras.Sequential([
    Conv2D(32, (5, 5), activation='relu', padding='same'),
    MaxPooling2D(pool_size=2, strides=2),
    Conv2D(64, (5, 5), activation='relu', padding='same'),
    MaxPooling2D(pool_size=2, strides=2),
    Flatten(),  # N*7*7*64 => N*3136
    layers.Dense(128, activation='tanh'),  # N*128
    layers.Dense(10, activation='softmax')])  # N*10

optimizer = optimizers.Adam(0.0001)


# ## 编译， fit以及evaluate

# In[4]:
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
train_ds, test_ds = mnist_dataset()
model.fit(train_ds, epochs=5)
model.evaluate(test_ds)
