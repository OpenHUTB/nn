#!/usr/bin/env python
# coding: utf-8

# ## 准备数据

# In[1]:


import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D,BatchNormalization

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

def mnist_dataset():# 准备MNIST数据集的主函数
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

def prepare_mnist_features_and_labels(x, y):# 预处理MNIST图像和标签的辅助函数
    x = tf.cast(x, tf.float32) / 255.0# 将像素值从[0, 255]归一化到[0, 1]，并转换为float32类型
    y = tf.cast(y, tf.int64)# 将标签转换为int64类型
    return x, y


# In[2]:


7*7*64


# ## 建立模型

# In[3]:


model = keras.Sequential([
    # 第一层卷积：使用3x3卷积核，增加特征图数量
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    BatchNormalization(),  # 添加批归一化
    MaxPooling2D(pool_size=2),
    
    # 第二层卷积：更深的网络结构
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=2),
    
    # 第三层卷积：进一步提取特征
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    
    Flatten(),
    # 使用ReLU激活函数和Dropout
    Dense(128, activation='relu'),
    Dropout(0.25),  # 减少过拟合
    Dense(10, activation='softmax')  # 输出层保持不变
])
    # 使用学习率调度器
lr_scheduler = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9
)
optimizer = optimizers.Adam(learning_rate=lr_scheduler)


# ## 编译， fit以及evaluate

# In[4]:


model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 添加早停和模型检查点
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        'best_mnist_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    )
]

# 准备数据集
train_ds, test_ds = mnist_dataset()

# 修正：从训练集中分割验证集，而不是测试集
train_size = int(0.9 * 60000)  # 假设原始训练集有60000个样本
val_ds = train_ds.take(6000).batch(100)  # 取6000个样本作为验证集
train_ds = train_ds.skip(6000)

# 训练模型
history = model.fit(
    train_ds,
    epochs=20,  # 增加轮次配合早停
    validation_data=val_ds,
    callbacks=callbacks
)

# 评估模型
test_loss, test_acc = model.evaluate(test_ds)
print(f"测试准确率: {test_acc * 100:.2f}%")

# 保存最终模型
model.save('mnist_model_final.h5')   


# In[ ]:





# In[ ]:




