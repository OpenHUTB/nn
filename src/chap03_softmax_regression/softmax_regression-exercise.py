#!/usr/bin/env python
# coding: utf-8

# # Softmax Regression Example

# ### 生成数据集， 看明白即可无需填写代码
# #### '<font color="blue">+</font>' 从高斯分布采样 (X, Y) ~ N(3, 6, 1, 1, 0).<br>
# #### '<font color="green">o</font>' 从高斯分布采样  (X, Y) ~ N(6, 3, 1, 1, 0)<br>
# #### '<font color="red">*</font>' 从高斯分布采样  (X, Y) ~ N(7, 7, 1, 1, 0)<br>

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt

from matplotlib import animation, rc
from IPython.display import HTML
import matplotlib.cm as cm
import numpy as np
# get_ipython().run_line_magic('matplotlib', 'inline')

dot_num = 100
epsilon = 1e-12  # 防止数值不稳定

# 生成蓝色类别的数据
x_p = np.random.normal(3., 1, dot_num)
y_p = np.random.normal(6., 1, dot_num)
y = np.ones(dot_num)  # 标签为1
C1 = np.array([x_p, y_p, y]).T

# 生成绿色类别的数据
x_n = np.random.normal(6., 1, dot_num)
y_n = np.random.normal(3., 1, dot_num)
y = np.zeros(dot_num)  # 标签为0
C2 = np.array([x_n, y_n, y]).T

# 生成红色类别的数据
x_b = np.random.normal(7., 1, dot_num)
y_b = np.random.normal(7., 1, dot_num)
y = np.ones(dot_num)*2  # 标签为2
C3 = np.array([x_b, y_b, y]).T

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

# 合并并打乱数据集
data_set = np.concatenate((C1, C2, C3), axis=0)
np.random.shuffle(data_set)


# ## 建立模型
# 建立模型类，定义loss函数，定义一步梯度下降过程函数

# In[1]:


class SoftmaxRegression():
    def __init__(self):
        '''============================='''
        # todo 填空一，构建模型所需的参数 self.W, self.b 可以参考logistic-regression-exercise
        self.W = tf.Variable(
            initial_value=tf.random.uniform([2, 3], minval=-0.1, maxval=0.1),
            dtype=tf.float32,
            name='weights'
        )
        self.b = tf.Variable(
            initial_value=tf.zeros([3]),
            dtype=tf.float32,
            name='bias'
        )
        '''============================='''

        self.trainable_variables = [self.W, self.b]

    @tf.function
    def __call__(self, inputs):
        # 计算logits并进行softmax归一化
        logits = tf.matmul(inputs, self.W) + self.b
        return tf.nn.softmax(logits)

@tf.function
def compute_loss(pred, label):
    # 将标签转换为one-hot编码
    label = tf.one_hot(tf.cast(label, tf.int32), depth=3, dtype=tf.float32)
    
    '''============================='''
    # 输入label shape(N,3), pred shape(N,3)
    # 输出 losses shape(N,) 每一个样本一个loss
    # todo 填空二，实现softmax的交叉熵损失函数(不使用tf内置的loss 函数)
    pred_clipped = tf.clip_by_value(pred, epsilon, 1. - epsilon)  # 防止log(0)
    losses = -tf.reduce_sum(label * tf.math.log(pred_clipped), axis=1)
    '''============================='''
    
    # 计算平均损失
    loss = tf.reduce_mean(losses)
    
    # 计算准确率
    pred_class = tf.argmax(pred, axis=1)
    label_class = tf.argmax(label, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_class, label_class), tf.float32))
    
    return loss, accuracy

@tf.function
def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss, acc = compute_loss(predictions, y)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, acc


# ### 实例化一个模型，进行训练

# In[12]:


if __name__ == '__main__':
    model = SoftmaxRegression()
    opt = tf.keras.optimizers.SGD(learning_rate=0.01)  # 使用随机梯度下降优化器
    
    # 数据预处理为Tensor
    x_data = data_set[:, :2].astype(np.float32)
    y_data = data_set[:, 2].astype(np.int32)
    x_tensor = tf.constant(x_data)
    y_tensor = tf.constant(y_data)
    
    # 训练配置
    num_epochs = 1000
    display_step = 50
    
    for epoch in range(num_epochs):
        loss, accuracy = train_one_step(model, opt, x_tensor, y_tensor)
        
        # 每display_step次迭代输出训练信息
        if (epoch + 1) % display_step == 0:
            print(f'Epoch {epoch+1}/{num_epochs} - Loss: {loss.numpy():.4f}, Accuracy: {accuracy.numpy():.4f}')

# ## 结果展示，无需填写代码

# In[13]:


# 创建网格进行预测
x_min, x_max = 0, 10
y_min, y_max = 0, 10
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)
grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

# 预测并绘制决策边界
Z = model(grid_points)
Z = np.argmax(Z.numpy(), axis=1).reshape(xx.shape)

# 绘制结果
plt.figure(figsize=(10,8))
plt.scatter(C1[:,0], C1[:,1], c='b', marker='+', label='Class 1')
plt.scatter(C2[:,0], C2[:,1], c='g', marker='o', label='Class 2')
plt.scatter(C3[:,0], C3[:,1], c='r', marker='*', label='Class 3')

# 绘制决策区域
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.colorbar(label='Predicted Class')
plt.title('Softmax Regression Decision Boundaries')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()