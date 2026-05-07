#!/usr/bin/env python
# coding: utf-8

# ## 准备数据

# In[1]:
# 导入操作系统相关功能，用于文件路径操作、环境变量等
import os

# 导入TensorFlow深度学习框架的核心功能
# tf提供了张量计算、自动微分、模型构建等基础功能
import tensorflow as tf

# 从TensorFlow中导入Keras高级API
# keras提供了简洁的模型定义、训练和评估接口
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
from tensorflow.keras.layers import Dense, Dropout, Flatten # 导入常用网络层：全连接层、正则化层和维度展平层
from tensorflow.keras.layers import Conv2D, MaxPooling2D  # 导入卷积层和最大池化层

#设置TensorFlow日志级别，只显示错误信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def mnist_dataset():
    """
    加载并预处理MNIST数据集，返回训练集和测试集的TensorFlow Dataset对象。

    Returns:
        ds (tf.data.Dataset): 处理后的训练数据集。
        test_ds (tf.data.Dataset): 处理后的测试数据集。
    """
    # 加载MNIST手写数字数据集
    # 返回格式：((训练图片, 训练标签), (测试图片, 测试标签))
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    x = x.reshape(x.shape[0], 28, 28, 1)# shape=(60000,28,28,1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)# shape=(10000,28,28,1)

    # 从NumPy数组创建TensorFlow Dataset对象
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.shuffle(60000).batch(100)  # 使用全部60000个训练样本

    # 创建测试数据集管道
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(prepare_mnist_features_and_labels)
    test_ds = test_ds.batch(512)  # 测试集10000张，按512分批
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
    x = tf.cast(x, tf.float32) / 255.0 # 归一化
    y = tf.cast(y, tf.int64) # 类型转换
    return x, y


# In[2]:
7 * 7 * 64 # 计算展平后的特征维度：7x7特征图，64个通道 -> 3136维向量


# 创建一个基于Keras Sequential API的卷积神经网络模型
# In[3]:
# 构建卷积神经网络（CNN）模型
model = keras.Sequential([
    # 第1层：卷积层（提取初级图像特征）
    Conv2D(32, (5, 5), activation='relu', padding='same'),
    # 批归一化：加速训练收敛，提高稳定性
    layers.BatchNormalization(),

    # 第2层：最大池化层（下采样，减少计算量）
    MaxPooling2D(pool_size=2, strides=2),

    # 第3层：卷积层（提取更高级特征）
    Conv2D(64, (5, 5), activation='relu', padding='same'),
    # 批归一化
    layers.BatchNormalization(),

    # 第4层：最大池化层（进一步下采样）
    MaxPooling2D(pool_size=2, strides=2),

    # 第5层：展平层（将3D特征图转换为1D向量）
    Flatten(),

    # Dropout正则化：训练时随机丢弃25%的神经元，防止过拟合
    Dropout(0.25),

    # 第6层：全连接层（特征整合）
    layers.Dense(128, activation='relu'),

    # Dropout正则化
    Dropout(0.5),

    # 第7层：输出层（分类预测）
    layers.Dense(10, activation='softmax')
])

optimizer = optimizers.Adam(0.001)

# ## 编译， fit以及evaluate

# In[4]:
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

train_ds, test_ds = mnist_dataset()

# 回调函数：早停（验证集loss不再下降时停止训练）+ 学习率衰减 + 保存最佳模型
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6
    ),
]

model.fit(train_ds, epochs=15, validation_data=test_ds, callbacks=callbacks)
model.evaluate(test_ds)
