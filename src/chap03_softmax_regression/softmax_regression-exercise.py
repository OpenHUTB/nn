#!/usr/bin/env python
# coding: utf-8

# # Softmax Regression Example - 优化版本

# ### 生成数据集， 看明白即可无需填写代码
# #### '<font color="blue">+</font>' 从高斯分布采样 (X, Y) ~ N(3, 6, 1, 1, 0).<br>
# #### '<font color="green">o</font>' 从高斯分布采样  (X, Y) ~ N(6, 3, 1, 1, 0)<br>
# #### '<font color="red">*</font>' 从高斯分布采样  (X, Y) ~ N(7, 7, 1, 1, 0)<br>

# In[1]:

# 导入运行所需模块
import tensorflow as tf # TensorFlow深度学习框架
import matplotlib.pyplot as plt # 数据可视化库
from matplotlib import animation, rc # 动画功能
import matplotlib.cm as cm # 颜色映射
import numpy as np # 数值计算库

# get_ipython().run_line_magic('matplotlib', 'inline')  # 仅在Jupyter环境下需要

# 设置数据点数量
dot_num = 500  # 增加数据量以提高模型性能

# 生成类别1的数据：均值为(3,6)，标准差为1
x_p = np.random.normal(3.0, 1, dot_num)  # x坐标
y_p = np.random.normal(6.0, 1, dot_num)  # y坐标
y = np.ones(dot_num)  # 标签为1
C1 = np.array([x_p, y_p, y]).T  # 组合成(x, y, label)格式

# 生成类别2的数据：均值为(6,3)，标准差为1
x_n = np.random.normal(6.0, 1, dot_num)
y_n = np.random.normal(3.0, 1, dot_num)
y = np.zeros(dot_num)  # 标签为0
C2 = np.array([x_n, y_n, y]).T

# 生成类别3的数据：均值为(7,7)，标准差为1
x_b = np.random.normal(7.0, 1, dot_num)
y_b = np.random.normal(7.0, 1, dot_num)
y = np.ones(dot_num) * 2  # 标签为2
C3 = np.array([x_b, y_b, y]).T

# 绘制三类样本的散点图
plt.scatter(C1[:, 0], C1[:, 1], c = "b", marker = "+")  # 类别1：蓝色加号
plt.scatter(C2[:, 0], C2[:, 1], c = "g", marker = "o")  # 类别2：绿色圆圈
plt.scatter(C3[:, 0], C3[:, 1], c = "r", marker = "*")  # 类别3：红色星号

# 合并所有类别的数据，形成完整数据集
data_set = np.concatenate((C1, C2, C3), axis=0)
np.random.shuffle(data_set)  # 随机打乱数据集顺序

# 数据预处理和分割
x1, x2, y = list(zip(*data_set))
X = np.array(list(zip(x1, x2)), dtype=np.float32)
y = np.array(y, dtype=np.int32)

# 自定义数据分割
indices = np.arange(len(X))
np.random.seed(42)
np.random.shuffle(indices)
n_train = int(len(X) * 0.7)
n_val = int(len(X) * 0.15)
train_idx = indices[:n_train]
val_idx = indices[n_train:n_train + n_val]
test_idx = indices[n_train + n_val:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

# 数据标准化
def standard_scale(X_train, X_val, X_test):
    """标准化数据"""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    # 避免除零
    std[std == 0] = 1.0

    X_train_scaled = (X_train - mean) / std
    X_val_scaled = (X_val - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_val_scaled, X_test_scaled, mean, std

X_train_scaled, X_val_scaled, X_test_scaled, feature_mean, feature_std = standard_scale(X_train, X_val, X_test)

print(f"训练集大小: {len(X_train)}")
print(f"验证集大小: {len(X_val)}")
print(f"测试集大小: {len(X_test)}")

# ## 建立模型
# 建立模型类，定义loss函数，定义一步梯度下降过程函数
#
# 填空一：在`__init__`构造函数中建立模型所需的参数
#
# 填空二：实现softmax的交叉熵损失函数(不使用tf内置的loss 函数)

# In[1]:

epsilon = 1e-7  # 优化数值稳定性

class SoftmaxRegression(tf.Module):
    def __init__(self, input_dim=2, num_classes=3, l2_reg_strength=0.01, dropout_rate=0.1):
        """
        初始化 Softmax 回归模型参数
        :param input_dim: 输入特征维度
        :param num_classes: 类别数量
        :param l2_reg_strength: L2正则化强度
        :param dropout_rate: dropout比率
        """
        super().__init__()
        self.l2_reg_strength = l2_reg_strength
        self.dropout_rate = dropout_rate

        # 使用He初始化方法改进
        self.W = tf.Variable(
            tf.random.truncated_normal([input_dim, num_classes], mean=0.0, stddev=0.1),
            name="W",
        )
        self.b = tf.Variable(tf.zeros([num_classes]), name="b")

    @tf.function
    def __call__(self, x, training=False):
        """
        模型前向传播：计算线性变换并应用softmax函数得到概率分布
        :param x: 输入数据，shape = (N, input_dim)
        :param training: 是否在训练模式下
        :return: softmax 概率分布，shape = (N, num_classes)
        """
        # 计算线性变换 logits
        logits = tf.matmul(x, self.W) + self.b

        # 添加dropout
        if training:
            logits = tf.nn.dropout(logits, rate=self.dropout_rate)

        # 应用softmax函数，将logits转换为概率分布
        return tf.nn.softmax(logits)

@tf.function
def compute_loss(pred, labels, model=None, num_classes=3):
    """
    计算交叉熵损失 + L2正则化 + 准确率
    :param pred: 模型输出，shape = (N, num_classes)
    :param labels: 实际标签，shape = (N,)
    :param model: 模型实例，用于L2正则化
    :param num_classes: 类别数
    :return: 平均损失值、准确率、不含正则化的损失
    """
    # 将真实标签转换为one-hot编码形式
    one_hot_labels = tf.one_hot(
        tf.cast(labels, tf.int32), depth=num_classes, dtype=tf.float32
    )

    # 直接使用预测概率计算交叉熵
    pred_clipped = tf.clip_by_value(pred, epsilon, 1.0)
    loss_without_reg = -tf.reduce_mean(
        tf.reduce_sum(one_hot_labels * tf.math.log(pred_clipped), axis=1)
    )

    # 计算L2正则化损失
    l2_loss = tf.constant(0.0, dtype=tf.float32)
    if model is not None:
        for var in model.trainable_variables:
            l2_loss += tf.nn.l2_loss(var)
    l2_loss *= model.l2_reg_strength if model is not None else 0.0

    # 总损失 = 交叉熵损失 + L2正则化损失
    loss = loss_without_reg + l2_loss

    # 计算准确率
    acc = tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.argmax(pred, axis=1),
                tf.argmax(one_hot_labels, axis=1)
            ),
            dtype=tf.float32,
        )
    )

    return loss, acc, loss_without_reg

@tf.function
def train_one_step(model, optimizer, x_batch, y_batch):
    # 单步训练：计算梯度并更新参数
    """
    一步梯度下降优化
    :param model: SoftmaxRegression 实例
    :param optimizer: 优化器（如 Adam, SGD）
    :param x_batch: 输入特征
    :param y_batch: 标签
    :return: 当前批次的损失、准确率、不含正则化的损失
    """
    # 使用 tf.GradientTape() 上下文管理器记录前向传播过程
    with tf.GradientTape() as tape:
        # 前向传播：计算模型对输入批次的预测（训练模式）
        predictions = model(x_batch, training=True)
        # 计算损失和准确率
        loss, accuracy, loss_without_reg = compute_loss(predictions, y_batch, model)

    # 计算损失函数对模型参数的梯度
    grads = tape.gradient(loss, model.trainable_variables)

    # 梯度裁剪，防止梯度爆炸
    grads, _ = tf.clip_by_global_norm(grads, clip_norm=5.0)

    # 优化步骤：使用优化器将计算出的梯度应用到模型参数上
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss, accuracy, loss_without_reg

# ### 实例化一个模型，进行训练，提取所需的数据

# In[12]:

model = SoftmaxRegression(l2_reg_strength=0.01, dropout_rate=0.1)
# 使用Adam优化器（比SGD更稳定）
opt = tf.keras.optimizers.Adam(learning_rate=0.01)

# 使用标准化后的训练数据进行训练
print("开始训练...")
best_val_acc = 0.0
patience = 50  # 早停耐心
wait = 0

for i in range(2000):  # 增加训练轮数
    # 训练步骤
    loss, accuracy, loss_without_reg = train_one_step(model, opt, X_train_scaled, y_train)

    # 验证集评估
    if i % 20 == 0:
        val_pred = model(X_val_scaled, training=False)
        val_loss, val_acc, _ = compute_loss(val_pred, y_val, model)
        val_loss = val_loss.numpy()
        val_acc = val_acc.numpy()

        print(f"Step {i:4d} | Train Loss: {loss.numpy():.4f} | Train Acc: {accuracy.numpy():.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # 早停机制
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at step {i}, best validation accuracy: {best_val_acc:.4f}")
                break

# 测试集评估
print("\n最终测试...")
test_pred = model(X_test_scaled, training=False)
test_loss, test_acc, _ = compute_loss(test_pred, y_test)
print(f"测试集 Loss: {test_loss.numpy():.4f} | 测试集 Acc: {test_acc.numpy():.4f}")

# # 结果展示，无需填写代码

# In[13]:

# 绘制三种不同类别的散点图
# C1[:, 0] 和 C1[:, 1] 分别表示 C1 的第一列和第二列数据（通常是特征）
plt.scatter(C1[:, 0], C1[:, 1], c="b", marker="+", s=80) # c="b" 设置颜色为蓝色，marker="+" 设置标记为加号
plt.scatter(C2[:, 0], C2[:, 1], c="g", marker="o", s=80) # c="g" 设置颜色为绿色，marker="o" 设置标记为圆形
plt.scatter(C3[:, 0], C3[:, 1], c="r", marker="*", s=80) # c="r" 设置颜色为红色，marker="*" 设置标记为星号

# 创建网格点用于绘制决策边界
x = np.arange(0.0, 10.0, 0.1)
y = np.arange(0.0, 10.0, 0.1)

# 生成网格坐标矩阵
# 将网格坐标展平并组合为输入特征矩阵
X, Y = np.meshgrid(x, y)
# 将X和Y数组重塑为一维数组后进行配对组合
inp = np.array(list(zip(X.reshape(-1), Y.reshape(-1))), dtype=np.float32)
print(inp.shape)
# 模型预测
Z = model(inp)
# 获取预测的类别
Z = np.argmax(Z, axis=1)
# 重塑为网络形状
Z = Z.reshape(X.shape)
# 绘制决策边界
plt.contour(X, Y, Z, alpha=0.5)
plt.show()

# 保存模型参数
ckpt = tf.train.Checkpoint(model=model) # 创建检查点
ckpt.write('softmax_regression_weights') # 保存模型

# 加载模型参数
ckpt.read('softmax_regression_weights')# 模型权重加载后即可用于新数据的多类别概率预测

# In[ ]:
