#!/usr/bin/env python
# coding: utf-8

# ## 设计基函数(basis function) 以及数据读取

# 导入必要的库
import numpy as np  # 数值计算库
import matplotlib.pyplot as plt  # 绘图库
import tensorflow as tf  # 深度学习框架
from tensorflow.keras import optimizers, layers, Model  # Keras API组件


# 基函数定义部分 ------------------------------------------------------------

def identity_basis(x):
    """恒等基函数：不对输入做任何变换，直接返回
    
    参数:
        x: 输入数据，形状为(N,)
        
    返回:
        形状为(N,1)的数组，即原始数据增加一个维度
    """
    return np.expand_dims(x, axis=1)  # 在axis=1维度增加一个维度，从(N,)变为(N,1)


def multinomial_basis(x, feature_num=10):
    """多项式基函数：生成多项式特征
    
    参数:
        x: 输入数据，形状为(N,)
        feature_num: 生成的特征数量，默认为10
        
    返回:
        形状为(N,feature_num)的数组，包含x的1次到feature_num次幂
    """
    x = np.expand_dims(x, axis=1)  # 形状从(N,)变为(N,1)
    feat = [x]  # 初始化特征列表，包含x^1
    
    # 生成x^2到x^feature_num
    for i in range(2, feature_num + 1):
        feat.append(x**i)
    
    # 沿axis=1拼接所有特征
    ret = np.concatenate(feat, axis=1)
    return ret


def gaussian_basis(x, feature_num=10):
    """高斯基函数：使用高斯分布作为基函数
    
    参数:
        x: 输入数据，形状为(N,)
        feature_num: 高斯函数的数量，默认为10
        
    返回:
        形状为(N,feature_num)的数组，每个特征对应一个高斯函数输出
    """
    # 在输入范围内均匀分布feature_num个中心点
    centers = np.linspace(0, 25, feature_num)
    # 计算高斯函数的宽度(标准差)，取相邻中心点间距
    width = 1.0 * (centers[1] - centers[0])
    
    # 将x从(N,)变为(N,1)再复制为(N,feature_num)
    x = np.expand_dims(x, axis=1)
    x = np.concatenate([x] * feature_num, axis=1)
    
    # 计算高斯函数值
    out = (x - centers) / width  # 标准化距离
    ret = np.exp(-0.5 * out ** 2)  # 高斯函数
    return ret


# 数据加载部分 ------------------------------------------------------------

def load_data(filename, basis_func=gaussian_basis):
    """载入数据并进行特征变换
    
    参数:
        filename: 数据文件名
        basis_func: 使用的基函数，默认为gaussian_basis
        
    返回:
        tuple: (变换后的数据(xs,ys), 原始数据(o_x,o_y))
    """
    xys = []
    with open(filename, "r") as f:
        for line in f:
            # 读取每行数据并转换为浮点数列表
            xys.append(list(map(float, line.strip().split())))
        
        # 将数据解压为特征和标签
        xs, ys = zip(*xys)
        # 转换为numpy数组
        xs, ys = np.asarray(xs), np.asarray(ys)
        
        # 保存原始数据
        o_x, o_y = xs, ys
        
        # 构造设计矩阵(design matrix)
        phi0 = np.expand_dims(np.ones_like(xs), axis=1)  # 偏置项(全1列)
        phi1 = basis_func(xs)  # 应用基函数变换
        
        # 拼接偏置项和变换后的特征
        xs = np.concatenate([phi0, phi1], axis=1)
        
        return (np.float32(xs), np.float32(ys)), (o_x, o_y)


# ## 定义模型 ------------------------------------------------------------

class linearModel(Model):
    """线性回归模型"""
    def __init__(self, ndim):
        """初始化模型
        
        参数:
            ndim: 输入特征的维度
        """
        super(linearModel, self).__init__()
        # 定义可训练参数w，形状为(ndim,1)
        self.w = tf.Variable(
            shape=[ndim, 1],
            initial_value=tf.random.uniform(
                [ndim, 1], minval=-0.1, maxval=0.1, dtype=tf.float32
            )
        )
        
    @tf.function  # 将Python函数编译为TensorFlow图，提高执行效率
    def call(self, x):
        """模型前向传播
        
        参数:
            x: 输入特征，形状为(batch_size, ndim)
            
        返回:
            预测值，形状为(batch_size,)
        """
        # 矩阵乘法后去掉多余的维度
        y = tf.squeeze(tf.matmul(x, self.w), axis=1)
        return y


# 数据加载和模型初始化
(xs, ys), (o_x, o_y) = load_data("train.txt")  # 加载训练数据
ndim = xs.shape[1]  # 获取特征维度
model = linearModel(ndim=ndim)  # 初始化模型


# ## 训练以及评估 ------------------------------------------------------------

# 使用Adam优化器，学习率0.1
optimizer = optimizers.Adam(0.1)


@tf.function
def train_one_step(model, xs, ys):
    """单步训练函数
    
    参数:
        model: 模型实例
        xs: 输入特征
        ys: 真实标签
        
    返回:
        当前步骤的损失值
    """
    # 在梯度带上下文中记录前向计算过程
    with tf.GradientTape() as tape:
        y_preds = model(xs)  # 前向传播
        loss = tf.keras.losses.MSE(ys, y_preds)  # 计算均方误差损失
        
    # 计算梯度
    grads = tape.gradient(loss, model.w)
    # 更新参数
    optimizer.apply_gradients([(grads, model.w)])
    return loss


@tf.function
def predict(model, xs):
    """预测函数
    
    参数:
        model: 模型实例
        xs: 输入特征
        
    返回:
        模型预测值
    """
    y_preds = model(xs)
    return y_preds


def evaluate(ys, ys_pred):
    """评估模型性能
    
    参数:
        ys: 真实值
        ys_pred: 预测值
        
    返回:
        预测值与真实值的标准差
    """
    std = np.std(ys - ys_pred)  # 计算残差的标准差
    return std


# 训练循环
for i in range(1000):  # 1000次迭代
    loss = train_one_step(model, xs, ys)
    if i % 100 == 1:  # 每100次打印一次损失
        print(f"loss is {loss:.4}")
                
# 训练集评估
y_preds = predict(model, xs)
std = evaluate(ys, y_preds)
print("训练集预测值与真实值的标准差：{:.1f}".format(std))

# 测试集评估
(xs_test, ys_test), (o_x_test, o_y_test) = load_data("test.txt")
y_test_preds = predict(model, xs_test)
std = evaluate(ys_test, y_test_preds)
print("测试集预测值与真实值的标准差：{:.1f}".format(std))


# 可视化结果 ------------------------------------------------------------

plt.plot(o_x, o_y, "ro", markersize=3)  # 绘制训练数据(红色圆点)
plt.plot(o_x_test, y_test_preds, "k")  # 绘制测试预测结果(黑色线)
plt.xlabel("x")  # x轴标签
plt.ylabel("y")  # y轴标签
plt.title("Linear Regression")  # 标题
plt.grid(True, linestyle="--", alpha=0.7, color="gray")  # 虚线网格
plt.legend(["train", "test", "pred"])  # 图例
plt.tight_layout()  # 自动调整子图参数
plt.show()  # 显示图像
