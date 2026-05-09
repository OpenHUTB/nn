#!/usr/bin/env python
# coding: utf-8
# # Logistic Regression Example - 优化版本
# ### 生成数据集，看明白即可无需填写代码
# #### '<font color="blue">+</font>' 从高斯分布采样 (X, Y) ~ N(3, 6, 1, 1, 0).<br>
# #### '<font color="green">o</font>' 从高斯分布采样 (X, Y) ~ N(6, 3, 1, 1, 0)<br>

# 导入 TensorFlow 深度学习框架
import tensorflow as tf
# 导入 matplotlib 的 pyplot 模块，用于数据可视化
import matplotlib.pyplot as plt

# 从 matplotlib 导入 animation 和 rc 模块
# animation：用于创建动态动画
# rc：运行时配置(runtime configuration)，用于设置图形默认参数
from matplotlib import animation
# 导入 IPython 的 HTML 显示功能，用于在 Notebook 中嵌入动画
# from IPython.display import HTML
# 导入 matplotlib 的 colormap 模块，用于颜色映射
# 导入 NumPy 数值计算库
import numpy as np

# 数据预处理和分割函数
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

def train_test_split_custom(X, y, test_size=0.3, val_size=0.15, random_state=42, stratify=None):
    """自定义数据分割"""
    if stratify is not None:
        # 分层采样
        from collections import Counter
        class_counts = Counter(stratify)
        indices_by_class = {cls: np.where(stratify == cls)[0] for cls in class_counts}

        train_idx, temp_idx = [], []
        for cls, indices in indices_by_class.items():
            n_train = int(len(indices) * (1 - test_size))
            np.random.seed(random_state)
            np.random.shuffle(indices)
            train_idx.extend(indices[:n_train])
            temp_idx.extend(indices[n_train:])
    else:
        indices = np.arange(len(X))
        np.random.seed(random_state)
        np.random.shuffle(indices)
        n_train = int(len(X) * (1 - test_size))
        train_idx = indices[:n_train]
        temp_idx = indices[n_train:]

    # 进一步分割验证集和测试集
    if val_size > 0:
        n_val = int(len(temp_idx) * (val_size / test_size))
        val_idx = temp_idx[:n_val]
        test_idx = temp_idx[n_val:]
        return train_idx, val_idx, test_idx

    return train_idx, temp_idx

# 设置随机种子（确保结果可复现）随机种子（Random Seed） 是用于初始化伪随机数生成器的起始值，它决定了随机数序列的"起点"。
# NumPy的随机种子
np.random.seed(42)
# TensorFlow的随机种子
tf.random.set_seed(42)

# 确保在 Jupyter Notebook 中内联显示图形
# get_ipython().run_line_magic('matplotlib', 'inline')

# 设置数据点数量
dot_num = 1000  # 增加数据量以提高模型性能
# 从均值为3，标准差为1的高斯分布中采样x坐标，用于正样本
x_p = np.random.normal(3., 1, dot_num)
# 从均值为6，标准差为1的高斯分布中采样y坐标，用于正样本
y_p = np.random.normal(6., 1, dot_num)
# 正样本的标签设为1
y = np.ones(dot_num)
# 将正样本的x、y坐标和标签组合成一个数组，形状为 (dot_num, 3)
C1 = np.array([x_p, y_p, y]).T
# random函数为伪随机数生成，并非真随机

# 从均值为6，标准差为1的高斯分布中采样x坐标，用于负样本
x_n = np.random.normal(6., 1, dot_num)
# 从均值为3，标准差为1的高斯分布中采样y坐标，用于负样本
y_n = np.random.normal(3., 1, dot_num)
# 负样本的标签设为0
y = np.zeros(dot_num)
# 将负样本的x、y坐标和标签组合成一个数组，形状为 (dot_num, 3)
C2 = np.array([x_n, y_n, y]).T

# 绘制正样本，用蓝色加号表示
plt.scatter(C1[:, 0], C1[:, 1], c='b', marker='+', alpha=0.6)
# 绘制负样本，用绿色圆圈表示
plt.scatter(C2[:, 0], C2[:, 1], c='g', marker='o', alpha=0.6)

# 将正样本和负样本连接成一个数据集
data_set = np.concatenate((C1, C2), axis=0)
# 随机打乱数据集的顺序
np.random.shuffle(data_set)

# 数据预处理和分割
x1, x2, y = list(zip(*data_set))
X = np.array(list(zip(x1, x2)), dtype=np.float32)
y = np.array(y, dtype=np.float32)

# 数据分割：训练集、验证集、测试集
train_idx, val_idx, test_idx = train_test_split_custom(X, y, test_size=0.3, val_size=0.15, random_state=42, stratify=y)

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

# 数据预处理：标准化（基于训练集统计量）
X_train_scaled, X_val_scaled, X_test_scaled, feature_mean, feature_std = standard_scale(X_train, X_val, X_test)

print(f"训练集大小: {len(X_train)}")
print(f"验证集大小: {len(X_val)}")
print(f"测试集大小: {len(X_test)}")

# ## 建立模型
# 建立模型类，定义loss函数，定义一步梯度下降过程函数
# 防止对数运算出现数值不稳定问题，添加一个极小值
epsilon = 1e-7  # 优化数值稳定性


class LogisticRegression():
    def __init__(self, input_dim=2, l2_reg_strength=0.01, dropout_rate=0.1):
        # 正则化常见目的：防止过拟合、提高稳定性、降低方差、提高泛化能力
        self.l2_reg_strength = l2_reg_strength
        self.dropout_rate = dropout_rate

        # 使用更优的He初始化方法
        self.W = tf.Variable(
            initial_value=tf.random.truncated_normal(
                shape=[input_dim, 1], mean=0.0, stddev=0.1
            )
        )
        # 初始化偏置变量b
        self.b = tf.Variable(
            shape=[1],
            dtype=tf.float32,
            initial_value=tf.zeros(shape=[1])
        )
        # 定义模型的可训练变量，即权重W和偏置b
        self.trainable_variables = [self.W, self.b]

    @tf.function
    def __call__(self, inp, training=False):
        """
           计算神经网络模型的前向传播过程

           参数:
               inp (tf.Tensor): 输入数据，形状通常为(N, D)，其中N是样本数，D是特征维度。
               training: 是否在训练模式下（用于dropout等）

           返回:
                tf.Tensor: 预测的概率值，形状为(N, 1)，值在[0, 1]之间。
        """
        # 计算输入数据与权重的矩阵乘法，再加上偏置，得到logits
        logits = tf.matmul(inp, self.W) + self.b

        # 添加dropout层以防止过拟合
        if training:
            logits = tf.nn.dropout(logits, rate=self.dropout_rate)

        # 对logits应用sigmoid函数，得到预测概率
        pred = tf.nn.sigmoid(logits)
        return pred


# 使用tf.function将该方法编译为静态图，提高执行效率
@tf.function
def compute_loss(pred, label, model=None):
    """
        计算二分类交叉熵损失 + L2正则化

        参数:
            pred: 模型预测的概率值
            label: 真实标签，取值为0或1
            model: 模型实例，用于计算L2正则化损失

        返回:
            loss: 平均交叉熵损失 + L2正则化项
            accuracy: 准确率
            loss_without_reg: 不含正则化的损失（用于监控）
    """
    if not isinstance(label, tf.Tensor):
        label = tf.constant(label, dtype=tf.float32)

    # 压缩预测结果的维度
    pred = tf.squeeze(pred, axis=1)

    # 计算交叉熵损失
    losses = -label * tf.math.log(pred + epsilon) - (1 - label) * tf.math.log(1 - pred + epsilon)
    loss_without_reg = tf.reduce_mean(losses)

    # 计算L2正则化损失
    l2_loss = tf.constant(0.0, dtype=tf.float32)

    # 总损失 = 交叉熵损失 + L2正则化损失
    loss = loss_without_reg + l2_loss

    # 计算准确率
    pred_labels = tf.where(pred > 0.5, tf.ones_like(pred), tf.zeros_like(pred))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(label, pred_labels), dtype=tf.float32))

    return loss, accuracy, loss_without_reg


@tf.function
def train_one_step(model, optimizer, x, y):
    # 使用GradientTape记录计算图，以便计算梯度
    with tf.GradientTape() as tape:
        # 使用模型对输入数据x进行预测
        pred = model(x)
        # 计算预测结果的损失和准确率
        loss, accuracy, loss_without_reg = compute_loss(pred, y)

    # 计算损失对可训练变量的梯度
    grads = tape.gradient(loss, model.trainable_variables)
    # 使用优化器更新模型的可训练变量
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, accuracy, loss_without_reg, model.W, model.b


if __name__ == '__main__':
    # 实例化逻辑回归模型
    # LogisticRegression 是一个用于二分类问题的线性模型
    model = LogisticRegression()

    # 使用自适应优化器 Adam ，学习率为0.01
    # Adam 是一种自适应学习率的优化算法，结合了 Momentum 和 RMSProp 的优点
    # learning_rate=0.01 设置了初始学习率为 0.01
    opt = tf.keras.optimizers.Adam(learning_rate = 0.01)  # 或Nadam/RMSprop

    # 从数据集中解包出x1, x2坐标和标签y
    # data_set 是一个包含多个样本的列表，每个样本格式为 (x1, x2, y)
    # 使用 zip(*data_set) 对数据进行转置，然后转换为列表
    # 这样可以将所有x1、x2和y分别分组
    x1, x2, y = list(zip(*data_set))

    # 将x1和x2组合成输入数据 x
    # 使用 zip(x1, x2) 将每个样本的x1和x2特征重新组合成特征对
    # 最终 x 的形式是 [(x1_1, x2_1), (x1_2, x2_2), ...]
    x = np.array(list(zip(x1, x2)), dtype = np.float32)
    y = np.array(y, dtype = np.float32)

    # 训练模型
    print("开始训练逻辑回归模型...")
    best_val_acc = 0.0
    patience = 30
    wait = 0

    # 使用标准化后的数据进行训练
    for i in range(500):  # 增加训练轮数
        loss, accuracy, loss_without_reg, W_opt, b_opt = train_one_step(model, opt, X_train_scaled, y_train)

        # 验证集评估
        if i % 20 == 0:
            val_pred = model(X_val_scaled, training=False)
            val_loss, val_acc, _ = compute_loss(val_pred, y_val, model)
            val_loss = val_loss.numpy()
            val_acc = val_acc.numpy()

            print(f"Step {i:3d} | Train Loss: {loss.numpy():.4f} | Train Acc: {accuracy.numpy():.4f} | "
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
    test_loss, test_acc, _ = compute_loss(test_pred, y_test, model)
    print(f"测试集 Loss: {test_loss.numpy():.4f} | 测试集 Acc: {test_acc.numpy():.4f}")

    # 用于动画可视化
    animation_frames = []
    for i in range(min(100, len(X_train_scaled))):  # 限制动画帧数
        sample_idx = np.random.randint(0, len(X_train_scaled), 50)
        sample_x, sample_y = X_train_scaled[sample_idx], y_train[sample_idx]
        loss, accuracy, _, W_opt, b_opt = train_one_step(model, opt, sample_x, sample_y)
        animation_frames.append((W_opt.numpy()[0, 0], W_opt.numpy()[1, 0], b_opt.numpy()[0], loss.numpy()))

    f, ax = plt.subplots(figsize=(6, 4))  # 创建一个图形和坐标轴
    f.suptitle('Logistic Regression Example', fontsize=15)  # 设置图形的标题
    plt.ylabel('Y')  # 设置Y轴标签为'Y'，用于标识垂直方向的变量
    plt.xlabel('X')  # 设置X轴标签为'X'，用于标识水平方向的变量
    ax.set_xlim(0, 10)  # X轴显示范围0-10
    ax.set_ylim(0, 10)  # Y轴显示范围0-10

    line_d, = ax.plot([], [], label = 'fit_line')  # 创建用于绘制决策边界的线条对象

    # 创建两个类别的数据点：
    # C1_dots：类别1的样本点，用蓝色'+'表示
    # C2_dots：类别2的样本点，用绿色'o'表示
    C1_dots, = ax.plot([], [], '+', c = 'b', label = 'actual_dots')   # 正样本
    C2_dots, = ax.plot([], [], 'o', c = 'g', label = 'actual_dots')   # 负样本

    # 创建用于显示动态文本的文本对象（位于左上角）
    # 参数说明：
    # 0.02, 0.95：文本位置坐标（x=2%轴宽度，y=95%轴高度），使用相对坐标系统
    # ''：初始空文本内容
    # horizontalalignment='left'：水平左对齐（使文本紧贴左侧边界）
    # verticalalignment='top'：垂直顶部对齐（使文本紧贴顶部边界）
    # transform=ax.transAxes：使用坐标轴相对坐标系（0-1范围，而非数据坐标系）
    frame_text = ax.text(
        0.02, 0.95, '',
        horizontalalignment='left',
        verticalalignment='top', 
        transform=ax.transAxes
    )

    # 决策边界为直线：W1·x1 + W2·x2 + b = 0，动态显示损失下降过程
    def init():
        """
        初始化动画所需的图形元素
        该函数将所有动态绘图对象的数据清空，为动画初始化做准备
        通常用于 Matplotlib 的 FuncAnimation 初始化函数
        """
        # 清空线条对象的数据（x, y 坐标）
        line_d.set_data([], [])
        # 清空类别1的散点数据（C1）
        C1_dots.set_data([], [])
        # 清空类别2的散点数据（C2）
        C2_dots.set_data([], [])
        # 返回所有需要动画更新的图形对象（打包为元组）
        # Matplotlib 动画要求返回值为 Artist 对象的集合
        return (line_d,) + (C1_dots,) + (C2_dots,)

    def animate(i):
        """
        动画的每一帧更新函数
        参数:
        i: 当前帧的索引
        返回:
        更新后的图形对象
        """
        # 具体实现 
        xx = np.arange(10, step=0.1)# 生成x轴数据点，范围0-9.9，步长0.1
        a = animation_frames[i][0]  # 从帧数据中提取当前帧的参数，假设animation_frames是一个列表，每个元素包含[a, b, c, loss]四个值
        b = animation_frames[i][1]  # 从帧数据中提取当前帧的参数b（通常表示偏移量或截距）
        c = animation_frames[i][2]  # 从帧数据中提取当前帧的参数c
        yy = a/-b * xx + c/-b       # 计算直线方程 y = (-a/b)x + (-c/b)
        line_d.set_data(xx, yy)     # 更新直线数据
        C1_dots.set_data(C1[:, 0], C1[:, 1]) # 更新C1和C2散点数据
        C2_dots.set_data(C2[:, 0], C2[:, 1]) # 更新C2散点数据
        frame_text.set_text(        # 更新帧文本信息，显示当前时间步和损失值
            'Timestep = %.1d/%.1d\nLoss = %.3f' % 
            (i, len(animation_frames), animation_frames[i][3])
        )
        return (line_d,) + (C1_dots,) + (C2_dots,)  # 返回需要更新的对象元组，用于blitting优化
    # 创建FuncAnimation对象
    anim = animation.FuncAnimation(
        f, animate, init_func=init, # 要绘制的图形对象，动画更新函数，初始化函数，设置动画初始状态
        frames=len(animation_frames), interval=50, blit=True, repeat=False # 帧间隔(毫秒)，是否使用blitting优化，# 是否循环播放
    )

    # from IPython.display import HTML
    # HTML(anim.to_html5_video())# 将动画转换为HTML5视频并显示
