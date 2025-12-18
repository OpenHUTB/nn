

# 卷积神经网络(CNN)



## 问题描述：

利用卷积神经网络，实现对MNIST 数据集的分类问题。




## 数据集: 

 	MNIST数据集包括60000张训练图片和10000张测试图片。图片样本的数量已经足够训练一个很复杂的模型（例如 CNN的深层神经网络）。它经常被用来作为一个新 的模式识别模型的测试用例。而且它也是一个方便学生和研究者们执行用例的数据集。除此之外，MNIST数据集是一个相对较小的数据集，可以在你的笔记本CPUs上面直接执行





## 题目要求： 

tensorflow版的卷积神经网路 conv2d() 和max_pool_2x2()函数，然后补全两层卷积的 8个空；

pytorch版本的卷积神经网络 需要补齐  self.conv1 中 nn.Conv2d( )  和 self.conv2( ) 的参数，还需要填写 x = x.view( )中的内容。

两个版本的训练精度都应该在 96% 以上。

---

## CNN_pytorch.py简介
本代码使用PyTorch实现了一个卷积神经网络（CNN），用于对MNIST手写数字数据集进行分类。代码包含数据加载、模型定义、训练和测试流程，适合初学者学习基本的深度学习流程和PyTorch框架的使用。

---

## 超参数说明
| 参数名           | 默认值 | 说明                      |
| ---------------- | ------ | ------------------------- |
| `learning_rate`  | 1e-4   | 优化器学习率              |
| `keep_prob_rate` | 0.7    | Dropout层保留神经元的比例 |
| `max_epoch`      | 3      | 训练轮数                  |
| `BATCH_SIZE`     | 50     | 每批训练数据量            |

---

## 模型结构
1. **卷积层1**：
   - 输入通道：1（灰度图）
   - 输出通道：32
   - 卷积核：7x7，步长1，填充3（保持尺寸）
   - 激活函数：ReLU
   - 池化：2x2最大池化（尺寸减半）

2. **卷积层2**：
   - 输入通道：32
   - 输出通道：64
   - 卷积核：5x5，步长1，填充2
   - 激活函数：ReLU
   - 池化：2x2最大池化

3. **全连接层**：
   - 输入：7x7x64 → 1024维
   - Dropout（保留比例0.7）
   - 输出：1024 → 10维（对应10个类别）

---

## 代码说明
- **数据加载**：自动下载MNIST数据集，分为训练集（60,000张）和测试集（500张样本用于验证）。
- **训练流程**：
  - 使用Adam优化器和交叉熵损失函数。
  - 每20个batch打印一次测试准确率。
- **测试流程**：计算模型在测试集上的分类准确率。

---

## 预期结果
- 训练3轮后，测试准确率约**98%~99%**（实际结果可能因随机性略有差异）。
- 输出示例：
  ```
  ========== 1000 ===== ===== 测试准确率:  0.966 ==========
  ========== 1020 ===== ===== 测试准确率:  0.974 ==========
  ========== 1040 ===== ===== 测试准确率:  0.974 ==========
  ========== 1060 ===== ===== 测试准确率:  0.976 ==========
  ========== 1080 ===== ===== 测试准确率:  0.98 ==========
  ========== 1100 ===== ===== 测试准确率:  0.968 ==========
  ========== 1120 ===== ===== 测试准确率:  0.974 ==========
  ========== 1140 ===== ===== 测试准确率:  0.976 ==========
  ========== 1160 ===== ===== 测试准确率:  0.974 ==========
  ========== 1180 ===== ===== 测试准确率:  0.968 ==========
  
  进程已结束，退出代码为 0
  ```

---
# CNN_tensorflow.py文档说明：

## 一、项目概述
本项目使用TensorFlow构建卷积神经网络（CNN），实现对MNIST手写数字数据集（0-9）的分类识别。通过多层卷积、池化和全连接层的组合，结合Dropout正则化技术，有效提升模型泛化能力，最终在测试集上达到99%以上的准确率。


## 二、环境依赖
```python
import tensorflow as tf          # 深度学习框架（建议2.x版本）
from tensorflow.examples.tutorials.mnist import input_data  # MNIST数据集加载工具
```


## 三、核心代码解析

### 1. 超参数设置
```python
learning_rate = 1e-4          # 学习率（控制梯度下降步长）
keep_prob_rate = 0.7          # Dropout保留概率（防止过拟合）
max_epoch = 2000              # 最大训练轮数
batch_size = 100              # 批量大小（每批训练样本数）
```


### 2. 关键函数定义
#### （1）权重与偏置初始化
```python
def weight_variable(shape):
    """使用截断正态分布初始化权重（标准差0.1）"""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """使用常数0.1初始化偏置（避免神经元静默）"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
```

#### （2）卷积与池化操作
```python
def conv2d(x, W, padding='SAME', strides=[1, 1, 1, 1]):
    """二维卷积层（支持SAME/VALID填充，自动验证参数合法性）"""
    if padding not in ['SAME', 'VALID']:
        raise ValueError("Padding must be 'SAME' or 'VALID'")
    return tf.nn.conv2d(x, W, strides=strides, padding=padding)

def max_pool_2x2(x):
    """2x2最大池化层（步长2，SAME填充）"""
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
```


### 3. 网络架构设计
#### （1）输入层
- 输入：784维向量（28x28像素图像展平），归一化至[0,1]
- 形状转换：`x_image = tf.reshape(xs, [-1, 28, 28, 1])`（转为4D张量：[批次, 高, 宽, 通道]）

#### （2）卷积层1
- 卷积核：7x7x1（输入通道1，输出通道32）
- 操作：卷积 + ReLU激活 + 2x2池化
- 输出形状：14x14x32（池化后尺寸减半，通道数增加）

#### （3）卷积层2
- 卷积核：5x5x32（输入通道32，输出通道64）
- 操作：卷积 + ReLU激活 + 2x2池化
- 输出形状：7x7x64

#### （4）全连接层1
- 输入：7x7x64展平为3136维向量
- 神经元：1024个，带ReLU激活和Dropout（保留率0.7）

#### （5）全连接层2（输出层）
- 神经元：10个（对应0-9分类）
- 激活函数：Softmax（输出概率分布）


### 4. 训练与评估
#### （1）损失函数与优化器
```python
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), axis=1))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
```
- 损失函数：交叉熵（适用于多分类问题）
- 优化器：Adam（自适应学习率优化算法）

#### （2）训练流程
```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(max_epoch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 批量训练
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: keep_prob_rate})
        if i % 100 == 0:
            acc = compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000])
            print(f"Iter {i}, Test Accuracy: {acc:.4f}")
```
- 每100轮迭代评估一次测试集准确率（前1000个样本）
- 典型输出：  
  `Iter 0, Test Accuracy: 0.1000`  
  `Iter 100, Test Accuracy: 0.9230`  
  `Iter 2000, Test Accuracy: 0.9910`


## 四、使用说明
### 1. 数据准备
- 数据集：MNIST（自动下载至`MNIST_data`目录）
- 数据格式：
  - 训练集：55,000样本（`mnist.train`）
  - 测试集：10,000样本（`mnist.test`）
  - 标签：One-Hot编码（如数字3对应[0,0,0,1,0,0,0,0,0,0]）

### 2. 运行步骤
1. 安装依赖：  
   ```bash
   pip install tensorflow==2.12.0
   ```
2. 执行脚本：  
   训练完成后，程序会自动输出每100轮的测试准确率。

### 3. 参数调整建议
| 参数          | 作用                          | 推荐调整范围       |
|---------------|-------------------------------|--------------------|
| `learning_rate` | 控制训练速度，防止梯度爆炸    | 1e-3 ~ 1e-5        |
| `keep_prob_rate` | Dropout保留率，调节正则强度   | 0.5 ~ 0.9          |
| `max_epoch`     | 训练轮数，避免过拟合          | 1000 ~ 5000        |
| `batch_size`    | 批量大小，影响内存占用        | 64 ~ 256           |


## 五、模型优化方向
1. **更深的网络结构**：  
   - 添加更多卷积层（如VGG-like架构）
   - 使用空洞卷积（Dilated Conv）增加感受野

2. **高级正则化技术**：  
   - 批量归一化（Batch Normalization）
   - L2正则化（`tf.nn.l2_loss`）

3. **数据增强**：  
   - 随机旋转、平移、缩放输入图像
   - 使用`tf.image`模块实现数据增强管道

4. **模型压缩**：  
   - 剪枝（Pruning）去除冗余连接
   - 量化（Quantization）降低权重精度


## 六、项目文件结构
```
mnist_cnn/
├── mnist_cnn.py       # 主程序文件
├── MNIST_data/        # 数据集存储目录
│   ├── train-images-idx3-ubyte.gz
│   └── ...（其他数据集文件）
└── README.md          # 项目说明文档
```
# tutorial_cifar10_conv-keras.py文档说明：


## 项目概述

本项目基于 TensorFlow 和 Keras 框架，实现了一个针对 CIFAR-10 数据集的 AlexNet 风格卷积神经网络模型。项目包含数据加载与预处理、模型构建、训练、评估以及特征可视化等功能。

## 功能特点

- **数据处理**：自动加载 CIFAR-10 数据集并进行标准化预处理
- **模型结构**：实现类似 AlexNet 的卷积神经网络架构
- **特征可视化**：可可视化模型各层的特征图输出
- **图像预测**：支持对自定义图像进行推理和特征提取

## 环境要求

### 依赖库

- Python 3.6+
- TensorFlow 2.0+
- Keras
- NumPy
- Matplotlib
- PIL (Pillow)

### 安装方法

```bash
# 创建并激活虚拟环境（可选但推荐）
python -m venv env
source env/bin/activate  # Linux/Mac
# 或
env\Scripts\activate  # Windows

# 安装依赖
pip install tensorflow numpy matplotlib pillow
```

## 数据集准备

### CIFAR-10 数据集

本项目使用 CIFAR-10 数据集，包含 10 个类别的 60000 张 32x32 彩色图像，其中 50000 张训练图像和 10000 张测试图像。

### 下载方式

1. 通过百度网盘下载：
   - 链接: https://pan.baidu.com/s/1LcCPcK9DgLS3W_DUPZS8kQ
   - 提取码: 5vwz

2. 解压后放置到 Keras 数据集目录：
   ```bash
   tar zxvf cifar***.tar.zip -C ~/.keras/datasets/
   ```

## 代码结构与功能说明

### 主要文件

- `cifar10_alexnet.py`：主程序文件，包含完整的模型实现和功能

### 核心功能模块

#### 1. 数据加载与预处理

```python
def cifar10_dataset():
    """加载并预处理CIFAR-10数据集，返回训练集和测试集的TensorFlow Dataset对象"""
    (x, y), (x_test, y_test) = datasets.cifar10.load_data()
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels).take(20000).shuffle(20000).batch(100)
    
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(prepare_mnist_features_and_labels).take(20000).batch(20000)
    return ds, test_ds

def prepare_mnist_features_and_labels(x, y):
    """预处理图像数据和标签，包括归一化和类型转换"""
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y
```

#### 2. 模型架构实现

```python
class MyConvModel(keras.Model):
    """实现类似AlexNet的卷积神经网络模型"""
    def __init__(self):
        super(MyConvModel, self).__init__()
        # 卷积层1：32个5x5卷积核，ReLU激活，same填充
        self.l1_conv = Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')
        # 卷积层2：64个5x5卷积核，ReLU激活，same填充
        self.l2_conv = Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same')
        # 最大池化层：2x2窗口，步长2
        self.pool = MaxPooling2D(pool_size=(2, 2), strides=2)
        # 展平层、全连接层
        self.flat = Flatten()
        self.dense1 = layers.Dense(100, activation='tanh')
        self.dense2 = layers.Dense(10)
    
    @tf.function
    def call(self, x):
        """前向传播过程"""
        h1 = self.l1_conv(x)
        h1_pool = self.pool(h1)
        h2 = self.l2_conv(h1_pool)
        h2_pool = self.pool(h2)
        flat_h = self.flat(h2_pool)
        dense1 = self.dense1(flat_h)
        logits = self.dense2(dense1)
        probs = tf.nn.softmax(logits, axis=-1)
        return probs
    
    @tf.function
    def getL1_feature_map(self, x):
        """获取第一层卷积的特征图"""
        return self.l1_conv(x)
    
    @tf.function
    def getL2_feature_map(self, x):
        """获取第二层卷积的特征图"""
        h1 = self.l1_conv(x)
        h1_pool = self.pool(h1)
        return self.l2_conv(h1_pool)
```

#### 3. 模型训练与评估

```python
# 编译模型
model.compile(
    optimizer=optimizers.Adam(0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型
train_ds, test_ds = cifar10_dataset()
model.fit(train_ds, epochs=10)

# 评估模型
model.evaluate(test_ds)
```

#### 4. 特征可视化与自定义图像处理

```python
# 加载并预处理自定义图像
img = Image.open(open('corgi.jpg', 'rb'))
img = numpy.asarray(img, dtype='float32') / 255.0  # 修正归一化系数
img = np.expand_dims(img, axis=0)  # 添加批次维度

# 获取第二层卷积的特征图
img_out = model.getL2_feature_map(img)

# 可视化原始图像和特征图
pylab.imshow(img[0, :, :, :])  # 显示原始图像

# 可视化前20个特征图
for i in range(0, 20, 4):
    pylab.figure(figsize=(10, 7))
    pylab.subplot(2, 2, 1); pylab.axis('off'); pylab.imshow(img_out[0, :, :, i])
    pylab.subplot(2, 2, 2); pylab.axis('off'); pylab.imshow(img_out[0, :, :, i+1])
    pylab.subplot(2, 2, 3); pylab.axis('off'); pylab.imshow(img_out[0, :, :, i+2])
    pylab.subplot(2, 2, 4); pylab.axis('off'); pylab.imshow(img_out[0, :, :, i+3])
    pylab.show()
```

## 使用方法

### 1. 运行训练与评估

```bash
python cifar10_alexnet.py
```

### 2. 自定义图像测试

1. 将待测试的图像（如`corgi.jpg`）放在与代码相同的目录下
2. 确保图像预处理部分的代码正确（注意归一化系数应为255而非256）
3. 运行代码，将自动显示原始图像和各层特征图

## 注意事项

1. **数据路径问题**：确保 CIFAR-10 数据集正确解压到 `~/.keras/datasets/` 目录
2. **图像预处理**：自定义图像需要调整尺寸至32x32像素，并注意归一化系数应为255
3. **特征可视化**：可视化代码会生成多个图像窗口，显示不同卷积层的特征图输出
4. **模型性能**：当前模型结构为简化版 AlexNet，实际准确率可能需要进一步优化

## 扩展与优化方向

1. **完整 AlexNet 架构**：添加更多卷积层和全连接层，匹配原始 AlexNet 结构
2. **数据增强**：添加随机裁剪、旋转、翻转等数据增强操作，提高模型泛化能力
3. **正则化技术**：加入 Dropout、L2 正则化等防止过拟合
4. **学习率调度**：实现动态学习率调整策略，优化训练过程
5. **模型优化**：调整卷积核数量、大小和网络深度，寻找最佳模型结构

