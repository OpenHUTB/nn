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

# 加载 MNIST 数据集并构建训练与测试集
def mnist_dataset():
    # 下载 MNIST 手写数字数据集，包含训练集和测试集
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    # 调整输入形状：添加通道维度（28, 28, 1）
    x = x.reshape(x.shape[0], 28, 28,1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28,1)
    
    # 创建训练数据集对象
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)  # 特征处理
    ds = ds.take(20000).shuffle(20000).batch(100)   # 随机打乱并分批

    # 创建测试数据集对象
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(prepare_mnist_features_and_labels)
    test_ds = test_ds.take(20000).shuffle(20000).batch(20000)  # 批量评估使用大批次
    return ds, test_ds

# 特征归一化并转换标签类型
def prepare_mnist_features_and_labels(x, y):
    x = tf.cast(x, tf.float32) / 255.0  # 像素值归一化到 0~1
    y = tf.cast(y, tf.int64)            # 标签转为 int64 类型
    return x, y


# ## 建立模型

# In[5]:

# 自定义卷积神经网络模型
class myConvModel(keras.Model):
    def __init__(self):
        super(myConvModel, self).__init__()
        # 第一层卷积，输出通道为32，卷积核为5x5，ReLU激活
        self.l1_conv = Conv2D(32, (5, 5), activation='relu', padding='same')
        # 第二层卷积，输出通道为64
        self.l2_conv = Conv2D(64, (5, 5), activation='relu', padding='same')
        # 最大池化层，步长为2，窗口为2x2
        self.pool = MaxPooling2D(pool_size=(2, 2), strides=2)
        # 扁平化层，将二维特征图展开为一维向量
        self.flat = Flatten()
        # 全连接层，输出维度为100，激活函数为 tanh
        self.dense1 = layers.Dense(100, activation='tanh')
        # 最终输出层，10类数字（不加 softmax）
        self.dense2 = layers.Dense(10)
   
    @tf.function  # 使用 tf.function 加速调用
    def call(self, x):
        h1 = self.l1_conv(x)        # 第一卷积层
        h1_pool = self.pool(h1)     # 第一次池化
        h2 = self.l2_conv(h1_pool)  # 第二卷积层
        h2_pool = self.pool(h2)     # 第二次池化
        flat_h = self.flat(h2_pool) # 扁平化
        dense1 = self.dense1(flat_h)# 全连接
        logits = self.dense2(dense1)# 输出层（logits）
        probs = tf.nn.softmax(logits, axis=-1)  # 应用 softmax 转为概率
        return probs

# 创建模型和优化器
model = myConvModel()
optimizer = optimizers.Adam()  # 使用 Adam 优化器


# ## 编译， fit以及evaluate

# In[6]:

# 编译模型，设置损失函数和评价指标
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',  # 稀疏交叉熵损失
              metrics=['accuracy'])                    # 评估准确率

# 获取训练集和测试集
train_ds, test_ds = mnist_dataset()

# 训练模型，训练5个 epoch
model.fit(train_ds, epochs=5)

# 在测试集上评估模型性能
model.evaluate(test_ds)


# In[ ]:




