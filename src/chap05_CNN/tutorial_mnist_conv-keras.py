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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

def mnist_dataset():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    x = x.reshape(x.shape[0], 28, 28,1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28,1)
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.take(20000).shuffle(20000).batch(100)


    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(prepare_mnist_features_and_labels)
    test_ds = test_ds.take(20000).shuffle(20000).batch(20000)
    return ds, test_ds

def prepare_mnist_features_and_labels(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y


# ## 建立模型

# In[5]:


class myConvModel(keras.Model):
    def __init__(self):
        super(myConvModel, self).__init__()
        self.l1_conv = Conv2D(32, (5, 5), activation='relu', padding='same')
        self.l2_conv = Conv2D(64, (5, 5), activation='relu', padding='same')
        self.pool = MaxPooling2D(pool_size=(2, 2), strides=2)
        self.flat = Flatten()
        self.dense1 = layers.Dense(100, activation='tanh')
        self.dense2 = layers.Dense(10)
    @tf.function
    def call(self, x):
        h1 = self.l1_conv(x)
        h1_pool = self.pool(h1)
        h2 = self.l2_conv(h1_pool)
        h2_pool = self.pool(h2)
        flat_h = self.flat(h2_pool)
        dense1 = self.dense1(flat_h)
        logits = self.dense2(dense1)
        probs = tf.nn.softmax(logits, axis=-1)
        return probs

model = myConvModel()
optimizer = optimizers.Adam()


# ## 编译， fit以及evaluate

# In[6]:


model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
train_ds, test_ds = mnist_dataset()
model.fit(train_ds, epochs=5)
model.evaluate(test_ds)


# In[ ]:




