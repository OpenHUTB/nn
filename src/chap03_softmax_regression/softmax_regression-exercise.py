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

# 设置数据点数量
dot_num = 100  
# 从均值为3，标准差为1的高斯分布中采样x坐标，用于正样本
x_p = np.random.normal(
    3.0, 1, dot_num
) 
# x和y坐标
y_p = np.random.normal(6.0, 1, dot_num)
# 标签为1
y = np.ones(dot_num)
# 组合成(x, y, label)格式
C1 = np.array([x_p, y_p, y]).T 

# 从均值为6，标准差为1的高斯分布中采样x坐标，用于负样本
x_n = np.random.normal(
    6.0, 1, dot_num
)
y_n = np.random.normal(3.0, 1, dot_num)
y = np.zeros(dot_num)
C2 = np.array([x_n, y_n, y]).T

# 从均值为7，标准差为1的高斯分布中采样x坐标，用于负样本
x_b = np.random.normal(
    7.0, 1, dot_num
)
y_b = np.random.normal(7.0, 1, dot_num)
y = np.ones(dot_num) * 2
C3 = np.array([x_b, y_b, y]).T

# 绘制正样本，用蓝色加号表示
plt.scatter(C1[:, 0], C1[:, 1], c="b", marker="+")
# 绘制负样本，用绿色圆圈表示
plt.scatter(C2[:, 0], C2[:, 1], c="g", marker="o")
# 绘制负样本，用红色星号表示
plt.scatter(C3[:, 0], C3[:, 1], c="r", marker="*")

# 将正样本和负样本连接成一个数据集
data_set = np.concatenate((C1, C2, C3), axis=0)
# 随机打乱数据集的顺序
np.random.shuffle(data_set)


# ## 建立模型
# 建立模型类，定义loss函数，定义一步梯度下降过程函数
#
# 填空一：在`__init__`构造函数中建立模型所需的参数
#
# 填空二：实现softmax的交叉熵损失函数(不使用tf内置的loss 函数)

# In[1]:


epsilon = 1e-12  # 防止 log(0)，处理数值稳定性问题


class SoftmaxRegression(tf.Module):
    def __init__(self, input_dim=2, num_classes=3):
        """
        初始化 Softmax 回归模型参数
        :param input_dim: 输入特征维度
        :param num_classes: 类别数量
        """
        super().__init__()
        # 初始化权重 W 和偏置 b
        # 使用均匀分布随机初始化
        self.W = tf.Variable(
            tf.random.uniform([input_dim, num_classes], minval=-0.1, maxval=0.1),
            name="W",
        )
        self.b = tf.Variable(tf.zeros([num_classes]), name="b")
        # 定义模型的可训练变量列表，用于梯度计算和参数更新
        self.trainable_variables = [self.W, self.b]

    @tf.function
    def __call__(self, x):
        """
        模型前向传播
        :param x: 输入数据，shape = (N, input_dim)
        :return: softmax 概率分布，shape = (N, num_classes)
        """
        #计算线性变换
        logits = tf.matmul(x, self.W) + self.b
        #应用softmax函数，将logits转换为概率分布
        return tf.nn.softmax(logits)


@tf.function
def compute_loss(pred, labels, num_classes=3):
    """
    计算交叉熵损失和准确率
    :param pred: 模型输出，shape = (N, num_classes)
    :param labels: 实际标签，shape = (N,)
    :param num_classes: 类别数
    :return: 平均损失值和准确率
    """
     # 将真实标签转换为one-hot编码形式
    one_hot_labels = tf.one_hot(
        tf.cast(labels, tf.int32), depth=num_classes, dtype=tf.float32
    )
    
    # 防止log(0)的情况，将预测概率限制在[epsilon, 1.0]范围内
    pred = tf.clip_by_value(pred, epsilon, 1.0)
    
    # 计算每个样本的交叉熵损失，对于每个样本，计算其真实类别的概率的负对数
    sample_losses = -tf.reduce_sum(one_hot_labels * tf.math.log(pred), axis=1)
    
    # 计算所有样本的平均损失
    loss = tf.reduce_mean(sample_losses)
    
    # 计算准确率，比较模型预测的类别和真实类别是否一致
    acc = tf.reduce_mean(
        tf.cast(
            tf.equal(tf.argmax(pred, axis=1), tf.argmax(one_hot_labels, axis=1)),
            dtype=tf.float32,
        )
    )
    
    return loss, acc


@tf.function
def train_one_step(model, optimizer, x_batch, y_batch):
    """
    一步梯度下降优化
    :param model: SoftmaxRegression 实例
    :param optimizer: 优化器（如 Adam, SGD）
    :param x_batch: 输入特征
    :param y_batch: 标签
    :return: 当前批次的损失与准确率
    """
    with tf.GradientTape() as tape:
        predictions = model(x_batch)
        loss, accuracy = compute_loss(predictions, y_batch)

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
# 转换为 float32
x = np.array(list(zip(x1, x2)), dtype=np.float32)  
# 转换为 int32
y = np.array(y, dtype=np.int32)  
# 从混合数据集 data_set 中提取特征和标签，并转换为所需的数据类型
for i in range(1000):
    loss, accuracy = train_one_step(model, opt, x, y)
    if i % 50 == 49:
        print(f"loss: {loss.numpy():.4}\t accuracy: {accuracy.numpy():.4}")
# 执行 1000 次迭代的模型训练，并每隔 50 步打印损失和准确率

# # 结果展示，无需填写代码

# In[13]:

# 绘制散点图
# C1[:, 0] 和 C1[:, 1] 分别表示 C1 的第一列和第二列数据（通常是特征）
plt.scatter(C1[:, 0], C1[:, 1], c="b", marker="+") # c="b" 设置颜色为蓝色，marker="+" 设置标记为加号
plt.scatter(C2[:, 0], C2[:, 1], c="g", marker="o")
plt.scatter(C3[:, 0], C3[:, 1], c="r", marker="*")

x = np.arange(0.0, 10.0, 0.1)
y = np.arange(0.0, 10.0, 0.1)


X, Y = np.meshgrid(x, y)
inp = np.array(list(zip(X.reshape(-1), Y.reshape(-1))), dtype=np.float32)
print(inp.shape)
Z = model(inp)
Z = np.argmax(Z, axis=1)
Z = Z.reshape(X.shape)
plt.contour(X, Y, Z)
plt.show()


# In[ ]:

