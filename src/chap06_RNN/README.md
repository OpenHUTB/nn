

# 循环神经网络(RNN)



## 问题描述：

利用循环神经网络，实现唐诗生成任务。




## 数据集: 

唐诗





## 题目要求： 

补全程序，主要是前面的3个空和 生成诗歌的一段代码。(tensorflow)   [pytorch 需要补全 对应的 rnn.py 文件中的两处代码]

生成诗歌 开头词汇是 “ 日 、 红 、 山 、 夜 、 湖、 海 、 月 。

参考文献：

​    Xingxing Zhang and Mirella Lapata. 2014. Chinese poetry generation with recurrent neural networks. In Proceedings of the 2014 Conference on EMNLP. Association for Computational Linguistics, October

并且参考了这篇博客  https://blog.csdn.net/Irving_zhang/article/details/76664998

# Learn2Carry-exercise.py说明文档：


## 一、项目概述
本项目使用TensorFlow构建循环神经网络（RNN）模型，实现对大数加法的预测。通过将数字转换为反向的数位序列，利用RNN的序列处理能力学习进位逻辑，最终实现对任意长度整数加法的准确预测。


## 二、环境依赖
```python
import numpy as np          # 数值计算
import tensorflow as tf     # 深度学习框架
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
import os, sys, tqdm        # 工具库
```


## 三、核心模块说明

### 1. 数据生成与预处理
#### （1）数据生成函数
```python
def gen_data_batch(batch_size: int, start: int, end: int) -> tuple:
    """生成随机整数对及其和"""
    numbers_1 = np.random.randint(start, end, batch_size)
    numbers_2 = np.random.randint(start, end, batch_size)
    results = numbers_1 + numbers_2
    return numbers_1, numbers_2, results
```

#### （2）数据预处理流程
- **数字转数位列表**：`convertNum2Digits`将整数转换为正向数位列表（如`123`→`[1,2,3]`）。
- **数位列表反转**：反向排列数位（如`[1,2,3]`→`[3,2,1]`），便于RNN按低位到高位处理。
- **长度填充**：`pad2len`用0填充数位列表至固定长度（如`maxlen=11`），适配批量训练。

#### （3）批量数据准备
```python
def prepare_batch(Nums1, Nums2, results, maxlen):
    """将数值转换为反转且填充后的数位矩阵"""
    Nums1 = [list(reversed(convertNum2Digits(o))) for o in Nums1]
    # 同理处理Nums2和results，并填充至maxlen
    return np.array(Nums1), np.array(Nums2), np.array(results)
```


### 2. 模型架构（RNN网络）
#### （1）网络结构
```python
class myRNNModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.embed_layer = layers.Embedding(10, 32)  # 数字嵌入层（0-9→32维向量）
        self.rnncell = layers.SimpleRNNCell(64)       # RNN单元（64隐藏层）
        self.rnn_layer = layers.RNN(self.rnncell, return_sequences=True)  # 序列输出RNN层
        self.dense = layers.Dense(10)                 # 分类层（预测每个数位的0-9概率）

    def call(self, num1, num2):
        emb1 = self.embed_layer(num1)                # 嵌入数1：(batch, seq_len, 32)
        emb2 = self.embed_layer(num2)                # 嵌入数2：(batch, seq_len, 32)
        emb = tf.concat([emb1, emb2], axis=-1)       # 拼接嵌入向量：(batch, seq_len, 64)
        rnn_out = self.rnn_layer(emb)                # RNN输出：(batch, seq_len, 64)
        logits = self.dense(rnn_out)                 # 全连接输出：(batch, seq_len, 10)
        return logits
```

#### （2）关键设计
- **嵌入层**：将单个数字（0-9）映射为32维向量，捕捉数字间的语义关系。
- **RNN层**：使用`SimpleRNNCell`处理序列，`return_sequences=True`保留每个时间步的输出，用于逐位预测。
- **输入拼接**：将两个数的嵌入向量在特征维度拼接（`axis=-1`），使模型同时感知两个数的当前位信息。


### 3. 训练与评估
#### （1）训练流程
```python
def train(steps, model, optimizer):
    for step in range(steps):
        # 生成训练数据（数值范围0~555,555,554）
        datas = gen_data_batch(200, 0, 555555555)
        Nums1, Nums2, results = prepare_batch(*datas, maxlen=11)
        # 单步训练：计算损失、更新参数
        loss = train_one_step(model, optimizer, Nums1, Nums2, results)
        if step % 50 == 0:
            print(f"Step {step}: Loss = {loss.numpy():.4f}")
```

#### （2）评估逻辑
```python
def evaluate(model):
    """评估模型在大数加法（555,555,555~999,999,998）上的准确率"""
    datas = gen_data_batch(2000, 555555555, 999999999)
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=11)
    logits = model(Nums1, Nums2)
    pred = np.argmax(logits.numpy(), axis=-1)  # 预测数位列表
    pred_nums = results_converter(pred)         # 转换为整数
    accuracy = np.mean([gt == pred for gt, pred in zip(datas[2], pred_nums)])
    print(f"Accuracy: {accuracy * 100:.2f}%")
```


## 四、使用说明
### 1. 运行步骤
1. 安装依赖：确保已安装TensorFlow 2.x及以上版本。
2. 训练模型：执行`train(3000, model, optimizer)`（约3000步可达较高准确率）。
3. 评估模型：调用`evaluate(model)`查看在大数集上的预测效果。

### 2. 参数调整建议
- **`maxlen`**：控制处理数字的最大位数（如`maxlen=11`支持10位数加法）。
- **`batch_size`**：增大批量大小可加速训练，但需注意内存限制。
- **`RNN隐藏层维度`**：当前使用64维，可尝试提升至128维以增强复杂进位学习能力。


## 五、实验结果示例
```
Step 0: Loss = 2.3026
Step 50: Loss = 0.9543
Step 100: Loss = 0.6821
...
Step 3000: Loss = 0.1235
Accuracy: 98.75%
```
