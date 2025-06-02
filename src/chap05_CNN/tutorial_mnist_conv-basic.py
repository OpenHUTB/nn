#!/usr/bin/env python
# coding: utf-8

# In[29]:
#导入了用于构建和训练卷积神经网络（CNN）的TensorFlow和Keras模块
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
#构建 MNIST 手写数字数据集的 TensorFlow 数据加载管道，包含数据预处理、批处理和随机打乱等操作
def mnist_dataset():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    x = x.reshape(x.shape[0], 28, 28,1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28,1)
    
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.take(20000).shuffle(20000).batch(32)
    
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(prepare_mnist_features_and_labels)
    test_ds = test_ds.take(20000).shuffle(20000).batch(20000)
    return ds, test_ds

def prepare_mnist_features_and_labels(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y


# In[ ]:

# ## 建立模型

# In[24]:


class myConvModel(keras.Model):
    def __init__(self):
        super(myConvModel, self).__init__()
        self.l1_conv = Conv2D(32, (5, 5), activation='relu', padding='same')#定义第一层卷积层,有32个过滤器,每个过滤器大小为5x5,使用ReLU激活函数
        self.l2_conv = Conv2D(64, (5, 5), activation='relu', padding='same')#第二层卷积层
        self.pool = MaxPooling2D(pool_size=(2, 2), strides=2)#池化层
        self.flat = Flatten() # 展平层：将多维张量转为一维向量
        self.dense1 = layers.Dense(100, activation='tanh')#全连接层,有100个神经元,使用tanh激活函数
        self.dense2 = layers.Dense(10)#输出层,有10个神经元
    @tf.function
    def call(self, x):
        #前向传播
        h1 = self.l1_conv(x)
        h1_pool = self.pool(h1)
        h2 = self.l2_conv(h1_pool)
        h2_pool = self.pool(h2)
        flat_h = self.flat(h2_pool)
        dense1 = self.dense1(flat_h)
        logits = self.dense2(dense1)
        return logits

model = myConvModel()
optimizer = optimizers.Adam()


# ## 定义loss以及train loop

# In[25]:


@tf.function
def compute_loss(logits, labels):
    #计算交叉熵损失
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))

@tf.function
def compute_accuracy(logits, labels):
    #计算预测准确率
    predictions = tf.argmax(logits, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))

@tf.function
def train_one_step(model, optimizer, x, y):

    with tf.GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(logits, y)

    # compute gradient(梯度)
    grads = tape.gradient(loss, model.trainable_variables)
    # update to weights(权重)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    accuracy = compute_accuracy(logits, y)

    # loss and accuracy is scalar tensor
    return loss, accuracy

@tf.function
def test_step(model, x, y):
    logits = model(x)
    loss = compute_loss(logits, y)
    accuracy = compute_accuracy(logits, y)
    return loss, accuracy

def train(epoch, model, optimizer, ds):
    loss = 0.0
    accuracy = 0.0
    for step, (x, y) in enumerate(ds):
        loss, accuracy = train_one_step(model, optimizer, x, y)

        if step % 500 == 0:
            print('epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())

    return loss, accuracy
def test(model, ds):
    loss = 0.0
    accuracy = 0.0
    for step, (x, y) in enumerate(ds):
        loss, accuracy = test_step(model, x, y)    
    print('test loss', loss.numpy(), '; accuracy', accuracy.numpy())
    return loss, accuracy

# # 训练

# In[26]:


train_ds, test_ds = mnist_dataset()
for epoch in range(2):
    loss, accuracy = train(epoch, model, optimizer, train_ds)
loss, accuracy = test(model, test_ds)


# In[ ]:





# In[ ]:




