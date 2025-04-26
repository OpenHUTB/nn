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
import numpy as np

# 设置随机种子保证可复现性
tf.random.set_seed(42)
np.random.seed(42)

dot_num = 100
epsilon = 1e-12  # 防止数值不稳定

# 数据生成部分优化
def generate_data():
    # 蓝色类别
    x_p = np.random.normal(3., 1, dot_num)
    y_p = np.random.normal(6., 1, dot_num)
    C1 = np.stack([x_p, y_p, np.ones(dot_num)], axis=1)
    
    # 绿色类别
    x_n = np.random.normal(6., 1, dot_num)
    y_n = np.random.normal(3., 1, dot_num)
    C2 = np.stack([x_n, y_n, np.zeros(dot_num)], axis=1)
    
    # 红色类别
    x_b = np.random.normal(7., 1, dot_num)
    y_b = np.random.normal(7., 1, dot_num)
    C3 = np.stack([x_b, y_b, np.ones(dot_num)*2], axis=1)
    
    return np.concatenate([C1, C2, C3], axis=0)

# 生成并打乱数据集
data_set = generate_data()
np.random.shuffle(data_set)

# 绘制初始数据分布
plt.figure(figsize=(8,6))
plt.scatter(data_set[:dot_num,0], data_set[:dot_num,1], c='b', marker='+', label='Class 1')
plt.scatter(data_set[dot_num:2*dot_num,0], data_set[dot_num:2*dot_num,1], c='g', marker='o', label='Class 2')
plt.scatter(data_set[2*dot_num:,0], data_set[2*dot_num:,1], c='r', marker='*', label='Class 3')
plt.title('Data Distribution')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()


# ## 建立模型
# 建立模型类，定义loss函数，定义一步梯度下降过程函数

# In[1]:


class SoftmaxRegression():
    def __init__(self):
        self.l2_lambda = 0.01  # L2正则化系数
        
        '''============================='''
        # todo 填空一，构建模型所需的参数 self.W, self.b 可以参考logistic-regression-exercise
        self.W = tf.Variable(
            initial_value=tf.random.truncated_normal([2, 3], stddev=0.1),
            regularizer=tf.keras.regularizers.L2(self.l2_lambda),
            name='weights'
        )
        self.b = tf.Variable(
            initial_value=tf.zeros([3]),
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
def compute_loss(model, pred, label):
    # 将标签转换为one-hot编码
    label = tf.one_hot(tf.cast(label, tf.int32), depth=3, dtype=tf.float32)
    
    '''============================='''
    # 输入label shape(N,3), pred shape(N,3)
    # 输出 losses shape(N,) 每一个样本一个loss
    # todo 填空二，实现softmax的交叉熵损失函数(不使用tf内置的loss 函数)
    pred_clipped = tf.clip_by_value(pred, epsilon, 1. - epsilon)  # 防止log(0)
    cross_entropy = -tf.reduce_sum(label * tf.math.log(pred_clipped), axis=1)
    losses = cross_entropy
    '''============================='''

    # 计算总损失（包含正则化项）
    total_loss = tf.reduce_mean(losses) + tf.reduce_sum(model.losses)
    
    # 计算准确率
    pred_class = tf.argmax(pred, axis=1)
    label_class = tf.argmax(label, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_class, label_class), tf.float32))
    
    return total_loss, accuracy

@tf.function
def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss, acc = compute_loss(model, predictions, y)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, acc


# ### 实例化一个模型，进行训练

# In[12]:


if __name__ == '__main__':
    model = SoftmaxRegression()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # 使用Adam优化器
    
    # 数据预处理
    x_data = data_set[:, :2].astype(np.float32)
    y_data = data_set[:, 2].astype(np.int32)
    
    # 使用TensorFlow Dataset API
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    dataset = dataset.shuffle(300).batch(32).prefetch(tf.data.AUTOTUNE)
    
    # 训练配置
    num_epochs = 1000
    display_step = 100
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        batch_count = 0
        
        for batch_x, batch_y in dataset:
            loss, acc = train_one_step(model, optimizer, batch_x, batch_y)
            epoch_loss += loss.numpy()
            epoch_acc += acc.numpy()
            batch_count += 1
            
        # 每display_step次迭代输出训练信息
        if (epoch + 1) % display_step == 0:
            avg_loss = epoch_loss / batch_count
            avg_acc = epoch_acc / batch_count
            print(f'Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}')

# ## 结果展示，无需填写代码

# In[13]:


# 创建网格进行预测
x_min, x_max = data_set[:,0].min()-1, data_set[:,0].max()+1
y_min, y_max = data_set[:,1].min()-1, data_set[:,1].max()+1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)
grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

# 预测并绘制决策边界
Z = model(grid_points)
Z = np.argmax(Z.numpy(), axis=1).reshape(xx.shape)

# 绘制结果
plt.figure(figsize=(12,9))
plt.scatter(data_set[:,0], data_set[:,1], c=data_set[:,2], 
            cmap='viridis', alpha=0.6, edgecolors='k', s=50)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis', levels=3)
plt.colorbar(label='Predicted Class')
plt.title('Softmax Regression with Decision Boundaries')
plt.xlabel('X')
plt.ylabel('Y')

# 添加类别标记
handles = [plt.Line2D([0], [0], marker='o', color='w', 
                     markerfacecolor=c, label=f'Class {i}', 
                     markersize=10) 
          for i,c in enumerate(['b','g','r'])]
plt.legend(handles=handles, loc='upper right')

plt.show()