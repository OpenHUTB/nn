#!/usr/bin/env python
# coding: utf-8

# ## 准备数据

# In[29]:


import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

def mnist_dataset():
    """
    加载并预处理MNIST数据集，返回训练集和测试集的Dataset对象
    
    返回:
    - ds: 训练数据集，批次大小32，包含20000个样本
    - test_ds: 测试数据集，批次大小20000，包含20000个样本
    """
    # 加载MNIST数据集
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    
    # 重塑数据为CNN所需的四维张量 [样本数, 高度, 宽度, 通道数]
    # MNIST图像为单通道灰度图，故通道数为1
    x = x.reshape(x.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # 创建训练数据集
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    # 应用数据预处理函数（归一化和类型转换）
    ds = ds.map(prepare_mnist_features_and_labels)
    # 限制训练集大小为20000个样本
    ds = ds.take(20000)
    # 打乱数据顺序，缓冲区大小为20000（覆盖全部样本）
    ds = ds.shuffle(20000)
    # 批处理数据，每批32个样本
    ds = ds.batch(32)
    
    # 创建测试数据集
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # 应用相同的数据预处理
    test_ds = test_ds.map(prepare_mnist_features_and_labels)
    # 限制测试集大小为20000个样本（原测试集有10000个，此处可能有冗余）
    test_ds = test_ds.take(20000)
    # 打乱测试数据（通常测试集不需要打乱，但此处保留）
    test_ds = test_ds.shuffle(20000)
    # 批处理测试数据，每批20000个样本（适合一次性评估）
    test_ds = test_ds.batch(20000)
    
    return ds, test_ds

def prepare_mnist_features_and_labels(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y


# In[ ]:





# ## 建立模型

# In[24]:


class myConvModel(keras.Model):
    """
    自定义卷积神经网络模型，用于图像分类任务
    
    网络结构：
    Conv2D(32) → Conv2D(64) → MaxPooling2D → Flatten → Dense(100) → Dense(10)
    """
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
        return logits

model = myConvModel()

optimizer = optimizers.Adam()


# ## 定义loss以及train loop

# In[25]:


@tf.function
def compute_loss(logits, labels):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits = logits, labels = labels))

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
    grads = tape.gradient(loss, model.trainable_variables)
    # update to weights
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




