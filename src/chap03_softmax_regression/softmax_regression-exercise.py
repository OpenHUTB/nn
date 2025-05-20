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
#get_ipython().run_line_magic('matplotlib', 'inline')

dot_num = 100 # 设置数据点数量
x_p = np.random.normal(3., 1, dot_num) # 从均值为3，标准差为1的高斯分布中采样x坐标，用于正样本
y_p = np.random.normal(6., 1, dot_num)
y = np.ones(dot_num)
C1 = np.array([x_p, y_p, y]).T

x_n = np.random.normal(6., 1, dot_num) # 从均值为6，标准差为1的高斯分布中采样x坐标，用于负样本
y_n = np.random.normal(3., 1, dot_num)
y = np.zeros(dot_num)
C2 = np.array([x_n, y_n, y]).T

x_b = np.random.normal(7., 1, dot_num) # 从均值为7，标准差为1的高斯分布中采样x坐标，用于负样本
y_b = np.random.normal(7., 1, dot_num)
y = np.ones(dot_num)*2
C3 = np.array([x_b, y_b, y]).T

plt.scatter(C1[:, 0], C1[:, 1], c='b', marker='+') # 绘制正样本，用蓝色加号表示
plt.scatter(C2[:, 0], C2[:, 1], c='g', marker='o') # 绘制负样本，用绿色圆圈表示
plt.scatter(C3[:, 0], C3[:, 1], c='r', marker='*') # 绘制负样本，用红色星号表示

data_set = np.concatenate((C1, C2, C3), axis=0) # 将正样本和负样本连接成一个数据集
np.random.shuffle(data_set) # 随机打乱数据集的顺序


# ## 建立模型
# 建立模型类，定义loss函数，定义一步梯度下降过程函数
# 
# 填空一：在`__init__`构造函数中建立模型所需的参数
# 
# 填空二：实现softmax的交叉熵损失函数(不使用tf内置的loss 函数)

# In[1]:


epsilon = 1e-12
class SoftmaxRegression():
    def __init__(self):
        '''============================='''
        #todo 填空一，构建模型所需的参数 self.W, self.b 可以参考logistic-regression-exercise
        self.W = tf.Variable(shape=[2, 3], dtype=tf.float32, 
                initial_value=tf.random.uniform(shape=[2, 3], minval=-0.1, maxval=0.1))
        self.b = tf.Variable(shape=[3], dtype=tf.float32, initial_value=tf.zeros(shape=[3]))
        '''============================='''
        
        self.trainable_variables = [self.W, self.b]
    @tf.function
    def __call__(self, inp):
        logits = tf.matmul(inp, self.W) + self.b # shape(N, 3)
        pred = tf.nn.softmax(logits)
        return pred    
    
@tf.function
def compute_loss(pred, label):
    label = tf.one_hot(tf.cast(label, dtype=tf.int32), dtype=tf.float32, depth=3)
    '''============================='''
    #输入label shape(N, 3), pred shape(N, 3)
    #输出 losses shape(N,) 每一个样本一个loss
    #todo 填空二，实现softmax的交叉熵损失函数(不使用tf内置的loss 函数)
    pred = tf.clip_by_value(pred, epsilon, 1.0)
    losses = -tf.reduce_sum(label * tf.math.log(pred), axis=1)
    '''============================='''
    loss = tf.reduce_mean(losses)
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(label,axis=1), tf.argmax(pred, axis=1)), dtype=tf.float32))
    return loss, accuracy

@tf.function
def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        pred = model(x)
        loss, accuracy = compute_loss(pred, y)
        
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, accuracy


# ### 实例化一个模型，进行训练

# In[12]:


model = SoftmaxRegression()
# 创建一个 SoftmaxRegression 模型实例 model
opt = tf.keras.optimizers.SGD(learning_rate=0.01)
# 创建随机梯度下降（SGD）优化器实例 opt，设置学习率为 0.01
x1, x2, y = list(zip(*data_set))
x = np.array(list(zip(x1, x2)), dtype=np.float32)  # 转换为 float32
y = np.array(y, dtype=np.int32)  # 转换为 int32
# 从混合数据集 data_set 中提取特征和标签，并转换为所需的数据类型
for i in range(1000):
    loss, accuracy = train_one_step(model, opt, x, y)
    if i%50==49:
        print(f'loss: {loss.numpy():.4}\t accuracy: {accuracy.numpy():.4}')
# 执行 1000 次迭代的模型训练，并每隔 50 步打印损失和准确率

# ## 结果展示，无需填写代码

# In[13]:


plt.scatter(C1[:, 0], C1[:, 1], c='b', marker='+')
plt.scatter(C2[:, 0], C2[:, 1], c='g', marker='o')
plt.scatter(C3[:, 0], C3[:, 1], c='r', marker='*')

x = np.arange(0., 10., 0.1)
y = np.arange(0., 10., 0.1)

X, Y = np.meshgrid(x, y)
inp = np.array(list(zip(X.reshape(-1), Y.reshape(-1))), dtype=np.float32)
print(inp.shape)
Z = model(inp)
Z = np.argmax(Z, axis=1)
Z = Z.reshape(X.shape)
plt.contour(X,Y,Z)
plt.show()


# In[ ]:




