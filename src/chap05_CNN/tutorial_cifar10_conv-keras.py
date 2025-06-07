#!/usr/bin/env python
# coding: utf-8
# # 参考 mnist_conv-keras 实现针对 cifar10 的 alexNet 卷积模型
# 
# 
# #### 链接: https://pan.baidu.com/s/1LcCPcK9DgLS3W_DUPZS8kQ   提取码: 5vwz
# ### 解压放到 ~/.keras/datasets/
# 
# ## tar zxvf cifar***.tar.zip

# ## 准备所需要的数据

# In[17]:
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy
import pylab
from PIL import Image
import numpy as np
import numpy
import pylab
from PIL import Image
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


def cifar10_dataset():
    """
    加载并预处理CIFAR-10数据集，返回训练集和测试集的TensorFlow Dataset对象
    
    返回:
        ds (tf.data.Dataset): 处理后的训练数据集
        test_ds (tf.data.Dataset): 处理后的测试数据集
    """
    # 加载CIFAR-10数据集
    # x: 图像数据 (50000张训练图像 + 10000张测试图像，32x32像素RGB)
    # y: 对应标签 (0-9的整数标签)
    (x, y), (x_test, y_test) = datasets.cifar10.load_data()
    # 创建训练数据集
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.take(20000).shuffle(20000).batch(100)

    # 创建测试数据集
    # 使用tf.data.Dataset.from_tensor_slices从内存中的NumPy数组创建数据集
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(prepare_mnist_features_and_labels)
    test_ds = test_ds.take(20000).batch(20000)
    return ds, test_ds

def prepare_mnist_features_and_labels(x, y):
    """
    预处理MNIST数据集的特征和标签
    
    参数:
    x: 图像数据，形状为 [样本数, 28, 28]，像素值范围 [0, 255]
    y: 标签数据，形状为 [样本数]，标签值范围 [0, 9]
    
    返回:
    x: 归一化后的图像数据，数据类型 float32，范围 [0, 1]
    y: 转换为int64类型的标签数据
    """
    # 将图像数据从uint8类型转换为float32类型
    # 并将像素值从[0, 255]归一化到[0, 1]范围
    # 归一化有助于梯度下降优化过程更稳定
    x = tf.cast(x, tf.float32) / 255.0
    
    # 将标签数据转换为int64类型
    # 这是TensorFlow中稀疏分类交叉熵损失函数要求的类型
    y = tf.cast(y, tf.int64)
    
    return x, y
# In[ ]:
# ## 开始建立模型

# In[18]:
class MyConvModel(keras.Model):
    '''在这里实现alexNet模型'''
    def __init__(self):
        """
        初始化自定义卷积神经网络模型。
        """
        super(MyConvModel, self).__init__()
        # 第一层卷积层
        self.l1_conv = Conv2D(filters=32, 
                              kernel_size=(5, 5), 
                              activation='relu', padding='same')
        # 第二层卷积层
        self.l2_conv = Conv2D(filters=64, 
                              kernel_size=(5, 5), 
                              activation='relu', padding='same')
        # 最大池化层
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
    
    @tf.function
    def getL1_feature_map(self, x):
        h1 = self.l1_conv(x) # [32, 28, 28, 32]
        
        return h1
    
    @tf.function
    def getL2_feature_map(self, x):
        h1 = self.l1_conv(x) # [32, 28, 28, 32]
        h1_pool = self.pool(h1) # [32, 14, 14, 32]
        h2 = self.l2_conv(h1_pool) #[32, 14, 14, 64]
        return h2

model = MyConvModel()
optimizer = optimizers.Adam(0.001)


# ## 编译， fit 以及 evaluate
# In[19]:
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
train_ds, test_ds = cifar10_dataset()
model.fit(train_ds, epochs=10)
model.evaluate(test_ds) #模型评估

# In[20]:


ds, test_ds = cifar10_dataset()

# 从测试数据集中获取第一个批次的第一张图像
for i in test_ds:
    test_batch = i[0][:1, :, :]  # 提取第一批中的第一张图像 [1, H, W, C]
    break                        # 只取一个样本，跳出循环

# 打开并预处理自定义图像（示例：柯基犬图片）
img = Image.open(open('corgi.jpg', 'rb'))  # 打开图像文件
img = numpy.asarray(img, dtype='float32')  # 转换为float32类型的numpy数组
img = img / 256.0                          # 错误：应除以255.0进行归一化
# print(img.shape)                         # 打印图像形状，例如 (224, 224, 3)

# 在第0维添加一个维度，将图像转换为批次格式 [1, H, W, C]
# 这是因为模型通常期望输入是批次形式的
img = np.expand_dims(img, axis=0)


# img = test_batch
img_out = model.getL2_feature_map(img)
pylab.imshow(img[0, :, :, :])# 显示原始输入图像（RGB）

pylab.figure(figsize=(10,7))
pylab.subplot(2, 2, 1); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 0])
pylab.subplot(2, 2, 2); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 1])
pylab.subplot(2, 2, 3); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 2])
pylab.subplot(2, 2, 4); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 3])
pylab.show()

pylab.figure(figsize=(10,7))
pylab.subplot(2, 2, 1); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 4])
pylab.subplot(2, 2, 2); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 5])
pylab.subplot(2, 2, 3); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 6])
pylab.subplot(2, 2, 4); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 7])
pylab.show()

pylab.figure(figsize=(10,7))
pylab.subplot(2, 2, 1); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 8])
pylab.subplot(2, 2, 2); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 9])
pylab.subplot(2, 2, 3); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 10])
pylab.subplot(2, 2, 4); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 11])
pylab.show()

pylab.figure(figsize=(10,7))
pylab.subplot(2, 2, 1); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 12])
pylab.subplot(2, 2, 2); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 13])
pylab.subplot(2, 2, 3); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 14])
pylab.subplot(2, 2, 4); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 15])
pylab.show()

pylab.figure(figsize=(10,7))
pylab.subplot(2, 2, 1); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 16])
pylab.subplot(2, 2, 2); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 17])
pylab.subplot(2, 2, 3); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 18])
pylab.subplot(2, 2, 4); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 19])
pylab.show()

# In[23]:

rand_model = MyConvModel()
ds, test_ds = cifar10_dataset()

for i in test_ds:
    test_batch = i[0][:1, :, :]
    break
img = Image.open(open('corgi.jpg', 'rb'))
img = numpy.asarray(img, dtype='float32') / 256.
# print(img.shape)
img = np.expand_dims(img, axis=0)

# img = test_batch
img_out = rand_model.getL1_feature_map(img)
pylab.imshow(img[0, :, :, :])

pylab.figure(figsize=(10,7))
pylab.subplot(2, 2, 1); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 0])
pylab.subplot(2, 2, 2); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 1])
pylab.subplot(2, 2, 3); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 2])
pylab.subplot(2, 2, 4); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 3])
pylab.show()

pylab.figure(figsize=(10,7))
pylab.subplot(2, 2, 1); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 4])
pylab.subplot(2, 2, 2); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 5])
pylab.subplot(2, 2, 3); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 6])
pylab.subplot(2, 2, 4); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 7])
pylab.show()

pylab.figure(figsize=(10,7))
pylab.subplot(2, 2, 1); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 8])
pylab.subplot(2, 2, 2); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 9])
pylab.subplot(2, 2, 3); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 10])
pylab.subplot(2, 2, 4); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 11])
pylab.show()

pylab.figure(figsize=(10,7))
pylab.subplot(2, 2, 1); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 12])
pylab.subplot(2, 2, 2); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 13])
pylab.subplot(2, 2, 3); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 14])
pylab.subplot(2, 2, 4); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 15])
pylab.show()

pylab.figure(figsize=(10,7))
pylab.subplot(2, 2, 1); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 16])
pylab.subplot(2, 2, 2); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 17])
pylab.subplot(2, 2, 3); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 18])
pylab.subplot(2, 2, 4); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 19])
pylab.show()

# In[ ]:
