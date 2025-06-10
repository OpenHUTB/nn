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

# 设置TensorFlow日志级别，避免输出过多信息
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
        
        self.flat = Flatten()   # 展平层：将多维特征张量展开为一维向量
        self.dense1 = layers.Dense(100, activation='tanh')
        self.dense2 = layers.Dense(10)
    @tf.function
    def call(self, x):
     # 第一层卷积
        h1 = self.l1_conv(x)  # 应用第一层卷积层
        h1_pool = self.pool(h1)  # 应用池化层

    # 第二层卷积
        h2 = self.l2_conv(h1_pool)  # 应用第二层卷积层
        h2_pool = self.pool(h2)  # 再次应用池化层

    # 展平
        flat_h = self.flat(h2_pool)  # 将多维张量展平为二维张量，形状为 [batch_size, num_features]

    # 全连接层
        dense1 = self.dense1(flat_h)  # 应用第一层全连接层

    # 输出层
        logits = self.dense2(dense1)  # 应用第二层全连接层，得到未归一化的 logits
        probs = tf.nn.softmax(logits, axis=-1)  # 应用 softmax 函数，将 logits 转换为概率分布
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
for i in test_ds:  # 遍历测试数据集（test_ds是一个可迭代对象）
    # 获取当前批次中的第一张图像
    # i[0]是图像数据，i[1]是对应的标签
    # [:1, :, :] 表示取第一个样本的所有高度、宽度和通道
    test_batch = i[0][:1, :, :]  # 结果形状为 [1, 高度, 宽度, 通道数]
    break  # 只需要一个样本，立即退出循环

# 打开并预处理自定义图像
# 以二进制读取模式打开图像文件
img = Image.open(open('corgi.jpg', 'rb'))  # 使用PIL库打开图像
# 将PIL图像转换为numpy数组，并指定数据类型为float32
img = numpy.asarray(img, dtype='float32')  # 形状通常为 (高度, 宽度, 通道数)
# 图像归一化处理（将像素值从0-255缩放到0-1范围）
img = img / 256.0  # 错误：应使用 img = img / 255.0
# 调试时可以打印图像形状查看维度信息
# print(img.shape)  

# 图像批次维度处理
# 在第0维添加一个维度，将图像从[H,W,C]转换为[1,H,W,C]格式
# 因为神经网络模型通常期望输入是批次形式的（即使只有一个图像）
img = np.expand_dims(img, axis=0)  # 添加批次维度后的形状：[1, 高度, 宽度, 通道数]

# img = test_batch
img_out = model.getL2_feature_map(img)
pylab.imshow(img[0, :, :, :])# 显示原始输入图像（RGB）

# 创建第一个图像展示画布：尺寸为10x7英寸（宽x高）
pylab.figure(figsize=(10,7))

# 第一行第一列子图：显示第0通道图像
pylab.subplot(2, 2, 1)  # 创建2行2列布局中的第1个子图
pylab.axis('off')       # 隐藏坐标轴，使图像更清晰
pylab.imshow(img_out[0, :, :, 0])  # 显示四维数组img_out中[0]批次、[0]通道的二维图像

# 第一行第二列子图：显示第1通道图像
pylab.subplot(2, 2, 2)  # 第2个子图位置
pylab.axis('off')
pylab.imshow(img_out[0, :, :, 1])  # 显示第1通道

# 第二行第一列子图：显示第2通道图像
pylab.subplot(2, 2, 3)  # 第3个子图位置
pylab.axis('off')
pylab.imshow(img_out[0, :, :, 2])  # 显示第2通道

# 第二行第二列子图：显示第3通道图像
pylab.subplot(2, 2, 4)  # 第4个子图位置
pylab.axis('off')
pylab.imshow(img_out[0, :, :, 3])  # 显示第3通道

pylab.show()  # 渲染并显示当前画布中的所有子图

# 重复上述过程，创建新画布展示后续通道
# 第二个画布：显示第4-7通道
pylab.figure(figsize=(10,7))
pylab.subplot(2, 2, 1); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 4])  # 通道4
pylab.subplot(2, 2, 2); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 5])  # 通道5
pylab.subplot(2, 2, 3); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 6])  # 通道6
pylab.subplot(2, 2, 4); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 7])  # 通道7
pylab.show()

# 第三个画布：显示第8-11通道
pylab.figure(figsize=(10,7))
pylab.subplot(2, 2, 1); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 8])   # 通道8
pylab.subplot(2, 2, 2); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 9])   # 通道9
pylab.subplot(2, 2, 3); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 10])  # 通道10
pylab.subplot(2, 2, 4); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 11])  # 通道11
pylab.show()

# 第四个画布：显示第12-15通道
pylab.figure(figsize=(10,7))
pylab.subplot(2, 2, 1); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 12])  # 通道12
pylab.subplot(2, 2, 2); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 13])  # 通道13
pylab.subplot(2, 2, 3); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 14])  # 通道14
pylab.subplot(2, 2, 4); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 15])  # 通道15
pylab.show()

# 第五个画布：显示最后4个通道（16-19）
pylab.figure(figsize=(10,7))
pylab.subplot(2, 2, 1); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 16])  # 通道16
pylab.subplot(2, 2, 2); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 17])  # 通道17
pylab.subplot(2, 2, 3); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 18])  # 通道18
pylab.subplot(2, 2, 4); pylab.axis('off'); pylab.imshow(img_out[0, :, :, 19])  # 通道19
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
