#!/usr/bin/env python
# coding: utf-8

# Logistic Regression Example
# ### 生成数据集，看明白即可无需填写代码
# #### '<font color="blue">+</font>' 从高斯分布采样 (X, Y) ~ N(3, 6, 1, 1, 0).
# #### '<font color="green">o</font>' 从高斯分布采样 (X, Y) ~ N(6, 3, 1, 1, 0).

import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
import matplotlib.cm as cm
import numpy as np

# 设置随机种子（确保结果可复现）
np.random.seed(42)
tf.random.set_seed(42)

# 确保在 Jupyter Notebook 中内联显示图形
get_ipython().run_line_magic('matplotlib', 'inline')

# 设置数据点数量
dot_num = 100

# 从均值为 3，标准差为 1 的高斯分布中采样 x 坐标，用于正样本
x_p = np.random.normal(3., 1, dot_num)
# 从均值为 6，标准差为 1 的高斯分布中采样 y 坐标，用于正样本
y_p = np.random.normal(6., 1, dot_num)
# 正样本的标签设为 1
y = np.ones(dot_num)
# 将正样本的 x、y 坐标和标签组合成一个数组，形状为 (dot_num, 3)
C1 = np.array([x_p, y_p, y]).T

# 从均值为 6，标准差为 1 的高斯分布中采样 x 坐标，用于负样本
x_n = np.random.normal(6., 1, dot_num)
# 从均值为 3，标准差为 1 的高斯分布中采样 y 坐标，用于负样本
y_n = np.random.normal(3., 1, dot_num)
# 负样本的标签设为 0
y = np.zeros(dot_num)
# 将负样本的 x、y 坐标和标签组合成一个数组，形状为 (dot_num, 3)
C2 = np.array([x_n, y_n, y]).T

# 绘制正样本，用蓝色加号表示
plt.scatter(C1[:, 0], C1[:, 1], c='b', marker='+')
# 绘制负样本，用绿色圆圈表示
plt.scatter(C2[:, 0], C2[:, 1], c='g', marker='o')

# 将正样本和负样本连接成一个数据集
data_set = np.concatenate((C1, C2), axis=0)
# 随机打乱数据集的顺序
np.random.shuffle(data_set)


class LogisticRegression:
    def __init__(self):
        l2_reg = tf.keras.regularizers.l2(0.01)
        self.W = tf.Variable(
            initial_value=tf.random.uniform(
                shape=[2, 1], minval=-0.1, maxval=0.1
            ),
            regularizer=l2_reg
        )
        self.b = tf.Variable(
            shape=[1],
            dtype=tf.float32,
            initial_value=tf.zeros(shape=[1])
        )
        self.trainable_variables = [self.W, self.b]

    @tf.function
    def __call__(self, inp):
        logits = tf.matmul(inp, self.W) + self.b
        pred = tf.nn.sigmoid(logits)
        return pred


epsilon = 1e-12


@tf.function
def compute_loss(pred, label):
    """计算二分类交叉熵损失函数。"""
    if not isinstance(label, tf.Tensor):
        label = tf.constant(label, dtype=tf.float32)
    pred = tf.squeeze(pred, axis=1)
    losses = -label * tf.math.log(pred + epsilon) - (1 - label) * tf.math.log(1 - pred + epsilon)
    loss = tf.reduce_mean(losses)
    pred = tf.where(pred > 0.5, tf.ones_like(pred), tf.zeros_like(pred))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(label, pred), dtype=tf.float32))
    return loss, accuracy


@tf.function
def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        pred = model(x)
        loss, accuracy = compute_loss(pred, y)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, accuracy, model.W, model.b


if __name__ == '__main__':
    model = LogisticRegression()
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    x1, x2, y = list(zip(*data_set))
    x = list(zip(x1, x2))
    animation_frames = []

    for i in range(200):
        loss, accuracy, W_opt, b_opt = train_one_step(model, opt, x, y)
        animation_frames.append((
            W_opt.numpy()[0, 0], W_opt.numpy()[1, 0], b_opt.numpy(), loss.numpy()
        ))
        if i % 20 == 0:
            print(f'loss: {loss.numpy():.4}\t accuracy: {accuracy.numpy():.4}')

    f, ax = plt.subplots(figsize=(6, 4))
    f.suptitle('Logistic Regression Example', fontsize=15)
    plt.ylabel('Y')
    plt.xlabel('X')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    line_d, = ax.plot([], [], label='fit_line')
    C1_dots, = ax.plot([], [], '+', c='b', label='actual_dots')
    C2_dots, = ax.plot([], [], 'o', c='g', label='actual_dots')
    frame_text = ax.text(
        0.02, 0.95, '',
        horizontalalignment='left',
        verticalalignment='top', 
        transform=ax.transAxes
    )

    def init():
        line_d.set_data([], [])
        C1_dots.set_data([], [])
        C2_dots.set_data([], [])
        return (line_d,) + (C1_dots,) + (C2_dots,)

    def animate(i):
        xx = np.arange(0, 10, 0.1)
        a = animation_frames[i][0]
        b = animation_frames[i][1]
        c = animation_frames[i][2]
        yy = (-a / b) * xx - c / b
        line_d.set_data(xx, yy)
        C1_dots.set_data(C1[:, 0], C1[:, 1])
        C2_dots.set_data(C2[:, 0], C2[:, 1])
        frame_text.set_text(
            'Timestep = %.1d/%.1d\nLoss = %.3f' % 
            (i, len(animation_frames), animation_frames[i][3])
        )
        return (line_d,) + (C1_dots,) + (C2_dots,)

    anim = animation.FuncAnimation(
        f, animate, init_func=init,
        frames=len(animation_frames), interval=30, blit=True
    )
    HTML(anim.to_html5_video())
