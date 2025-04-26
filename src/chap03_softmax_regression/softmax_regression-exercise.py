#!/usr/bin/env python
# coding: utf-8

# # Softmax Regression Example

# ### 数据集生成模块
# 生成三维数据集（X,Y坐标+类别标签），包含三个高斯分布的二维点集

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt

from matplotlib import animation, rc
from IPython.display import HTML
import matplotlib.cm as cm
import numpy as np

dot_num = 100
epsilon = 1e-12  # 防止log(0)或概率为0时的数值溢出

# 数据生成与可视化模块
def generate_data():
    # 蓝色类别（+标记）
    x_p = np.random.normal(3., 1, dot_num)  # 均值3，标准差1的正态分布
    y_p = np.random.normal(6., 1, dot_num)
    y = np.ones(dot_num)  # 标签1对应蓝色
    C1 = np.array([x_p, y_p, y]).T  # 转置后形状为(100,3)
    
    # 绿色类别（o标记）
    x_n = np.random.normal(6., 1, dot_num)  # 均值6
    y_n = np.random.normal(3., 1, dot_num)
    y = np.zeros(dot_num)  # 标签0对应绿色
    C2 = np.array([x_n, y_n, y]).T
    
    # 红色类别（*标记）
    x_b = np.random.normal(7., 1, dot_num)  # 均值7
    y_b = np.random.normal(7., 1, dot_num)
    y = np.ones(dot_num)*2  # 标签2对应红色
    C3 = np.array([x_b, y_b, y]).T
    
    return C1, C2, C3

# 生成并可视化数据
C1, C2, C3 = generate_data()

# 绘制初始数据分布
plt.figure(figsize=(8,6))
plt.scatter(C1[:,0], C1[:,1], c='b', marker='+', label='Class 1')
plt.scatter(C2[:,0], C2[:,1], c='g', marker='o', label='Class 2')
plt.scatter(C3[:,0], C3[:,1], c='r', marker='*', label='Class 3')
plt.title('Data Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# 数据预处理
data_set = np.concatenate((C1, C2, C3), axis=0)
np.random.shuffle(data_set)  # 打乱数据集顺序
print(f"数据集形状：{data_set.shape}")  # 验证数据集结构

# ## 模型构建模块
# 实现Softmax回归模型及训练流程

# In[1]:


class SoftmaxRegression():
    def __init__(self):
        # 模型参数初始化
        self.W = tf.Variable(
            initial_value=tf.random.uniform([2, 3], minval=-0.1, maxval=0.1),  # 2特征→3类别的权重矩阵
            dtype=tf.float32,
            name='weights'
        )
        self.b = tf.Variable(
            initial_value=tf.zeros([3]),  # 偏置项初始化为零向量
            dtype=tf.float32,
            name='bias'
        )
        self.trainable_variables = [self.W, self.b]  # 定义可训练参数列表

    @tf.function
    def __call__(self, inputs):
        """前向传播计算：线性变换+Softmax归一化"""
        logits = tf.matmul(inputs, self.W) + self.b  # 线性组合：XW + b
        return tf.nn.softmax(logits)  # 转换为概率分布（每行和为1）

@tf.function
def compute_loss(pred, label):
    """实现交叉熵损失函数"""
    label = tf.one_hot(tf.cast(label, tf.int32), depth=3, dtype=tf.float32)  # 转换为one-hot编码
    
    # 防止数值不稳定
    pred_clipped = tf.clip_by_value(pred, epsilon, 1 - epsilon)  
    cross_entropy = -tf.reduce_sum(label * tf.math.log(pred_clipped), axis=1)  # 每样本交叉熵
    
    # 计算平均损失和准确率
    loss = tf.reduce_mean(cross_entropy)
    pred_class = tf.argmax(pred, axis=1)
    label_class = tf.argmax(label, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_class, label_class), tf.float32))
    
    return loss, accuracy

@tf.function
def train_one_step(model, optimizer, x, y):
    """单次梯度更新步骤"""
    with tf.GradientTape() as tape:
        predictions = model(x)  # 前向传播
        loss, acc = compute_loss(predictions, y)  # 计算损失和准确率
        
    gradients = tape.gradient(loss, model.trainable_variables)  # 计算梯度
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # 更新参数
    return loss, acc


# ### 模型训练与评估

# In[12]:


if __name__ == '__main__':
    model = SoftmaxRegression()
    opt = tf.keras.optimizers.SGD(learning_rate=0.01)  # 随机梯度下降优化器
    
    # 数据转换为Tensor
    x_data = data_set[:, :2].astype(np.float32)
    y_data = data_set[:, 2].astype(np.int32)
    x_tensor = tf.constant(x_data)
    y_tensor = tf.constant(y_data)
    
    # 训练配置
    num_epochs = 1000
    display_step = 50
    
    # 训练循环
    for epoch in range(num_epochs):
        loss, accuracy = train_one_step(model, opt, x_tensor, y_tensor)
        
        # 进度输出
        if (epoch + 1) % display_step == 0:
            print(f'Epoch {epoch+1}/{num_epochs} - Loss: {loss.numpy():.4f}, Accuracy: {accuracy.numpy():.4f}')

# ## 可视化结果模块

# In[13]:


# 决策边界绘制
x_min, x_max = 0, 10
y_min, y_max = 0, 10
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),  # 生成200个x坐标点
    np.linspace(y_min, y_max, 200)   # 生成200个y坐标点
)
grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)  # 展开为二维点集

# 预测所有网格点类别
Z = model(grid_points)
Z = np.argmax(Z.numpy(), axis=1).reshape(xx.shape)  # 转换为二维类别矩阵

# 绘制结果
plt.figure(figsize=(10,8))
plt.scatter(C1[:,0], C1[:,1], c='b', marker='+', label='Class 1')
plt.scatter(C2[:,0], C2[:,1], c='g', marker='o', label='Class 2')
plt.scatter(C3[:,0], C3[:,1], c='r', marker='*', label='Class 3')

# 绘制决策区域
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis', levels=3)  # 3个类别区域
plt.colorbar(label='Predicted Class')
plt.title('Softmax Regression Decision Boundaries')
plt.xlabel('X')
plt.ylabel('Y')

# 添加图例说明
handles = [plt.Line2D([0], [0], marker='o', color='w', 
                     markerfacecolor=c, label=f'Class {i}', 
                     markersize=10) 
          for i,c in enumerate(['b','g','r'])]
plt.legend(handles=handles, loc='upper right')

plt.show()