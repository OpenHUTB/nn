<<<<<<< HEAD


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
=======
# 循环神经网络（RNN）实验说明

## 一、项目简介

本项目围绕循环神经网络（RNN）的基本原理与应用展开，主要包含两个实验任务：

1. **唐诗生成任务**  
   利用循环神经网络学习唐诗文本的序列特征，并根据指定开头词生成诗句。

2. **大数加法预测任务**  
   使用 TensorFlow 构建 RNN 模型，通过学习数字序列中的进位规律，实现对大数加法结果的预测。

本项目旨在帮助理解 RNN 在序列建模、文本生成和数值序列学习中的基本应用。

---

## 二、实验任务一：唐诗生成

### 1. 问题描述

利用循环神经网络实现唐诗自动生成任务。模型通过学习唐诗数据集中词语或字符之间的上下文关系，生成符合一定语言规律的诗句。

### 2. 数据集

数据集使用唐诗文本数据。

### 3. 题目要求

需要补全程序中的相关代码：

- TensorFlow 版本：主要补全前 3 个空以及生成诗歌部分代码；
- PyTorch 版本：需要补全 `rnn.py` 文件中的两处代码。

生成诗歌时，要求使用以下词语作为开头：

```text
日、红、山、夜、湖、海、月
### 4. 参考资料
Xingxing Zhang and Mirella Lapata. 2014. Chinese Poetry Generation with Recurrent Neural Networks. Proceedings of EMNLP 2014.
参考博客：https://blog.csdn.net/Irving_zhang/article/details/76664998
## 三、实验任务二：Learn2Carry 大数加法预测
### 1. 项目概述

Learn2Carry-exercise.py 使用 TensorFlow 构建循环神经网络模型，实现对大数加法结果的预测。

该任务将整数拆分为数位序列，并将数位顺序反转，使模型可以从低位到高位依次学习加法中的进位规律。通过训练，RNN 可以逐位预测两个整数相加后的结果。

## 四、环境依赖

运行本项目需要安装以下依赖：

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
import os
import sys
import tqdm

建议环境：

Python >= 3.8
TensorFlow >= 2.x
NumPy
tqdm
## 五、核心模块说明
### 1. 数据生成

通过随机生成两个整数及其求和结果，构造训练数据。

def gen_data_batch(batch_size: int, start: int, end: int) -> tuple:
    """生成随机整数对及其求和结果"""
>>>>>>> upstream/main
    numbers_1 = np.random.randint(start, end, batch_size)
    numbers_2 = np.random.randint(start, end, batch_size)
    results = numbers_1 + numbers_2
    return numbers_1, numbers_2, results
<<<<<<< HEAD
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
=======
### 2. 数据预处理

为了让 RNN 更容易学习加法中的进位规律，需要对整数进行如下处理：

整数转数位列表
例如：

123 -> [1, 2, 3]

反转数位顺序
例如：

[1, 2, 3] -> [3, 2, 1]

这样可以让 RNN 从个位开始处理，更符合加法进位的计算顺序。

长度填充
使用 0 将所有数位序列填充到统一长度，便于批量训练。
def prepare_batch(Nums1, Nums2, results, maxlen):
    """将整数转换为反转并填充后的数位矩阵"""
    Nums1 = [list(reversed(convertNum2Digits(o))) for o in Nums1]
    Nums2 = [list(reversed(convertNum2Digits(o))) for o in Nums2]
    results = [list(reversed(convertNum2Digits(o))) for o in results]

    Nums1 = [pad2len(o, maxlen) for o in Nums1]
    Nums2 = [pad2len(o, maxlen) for o in Nums2]
    results = [pad2len(o, maxlen) for o in results]

    return np.array(Nums1), np.array(Nums2), np.array(results)
### 3. RNN 模型结构

模型主要由以下部分组成：

Embedding 层：将数字 0-9 映射为向量表示；
RNN 层：学习数位序列中的进位规律；
Dense 层：对每个时间步输出 0-9 的分类结果。
class myRNNModel(keras.Model):
    def __init__(self):
        super().__init__()

        self.embed_layer = layers.Embedding(10, 32)
        self.rnncell = layers.SimpleRNNCell(64)
        self.rnn_layer = layers.RNN(self.rnncell, return_sequences=True)
        self.dense = layers.Dense(10)

    def call(self, num1, num2):
        emb1 = self.embed_layer(num1)
        emb2 = self.embed_layer(num2)

        emb = tf.concat([emb1, emb2], axis=-1)
        rnn_out = self.rnn_layer(emb)
        logits = self.dense(rnn_out)

        return logits
### 4. 模型设计说明
Embedding 层

将单个数字映射为高维向量，使模型能够学习数字之间的表示关系。

RNN 层

使用 SimpleRNNCell 处理序列数据，并设置 return_sequences=True，使模型在每一个时间步都输出预测结果。

特征拼接

两个加数的数位嵌入会在最后一个维度进行拼接，使模型能够同时学习两个数当前位之间的关系。

## 六、训练与评估
### 1. 训练流程

训练过程中，每一步都会随机生成一批整数对，并将其转换为数位序列输入模型。

def train(steps, model, optimizer):
    for step in range(steps):
        datas = gen_data_batch(200, 0, 555555555)
        Nums1, Nums2, results = prepare_batch(*datas, maxlen=11)

        loss = train_one_step(model, optimizer, Nums1, Nums2, results)

        if step % 50 == 0:
            print(f"Step {step}: Loss = {loss.numpy():.4f}")
### 2. 模型评估

评估阶段会在更大的数值范围上测试模型泛化能力。

def evaluate(model):
    """评估模型在大数加法任务上的准确率"""
    datas = gen_data_batch(2000, 555555555, 999999999)
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=11)

    logits = model(Nums1, Nums2)
    pred = np.argmax(logits.numpy(), axis=-1)

    pred_nums = results_converter(pred)
    accuracy = np.mean([gt == pred for gt, pred in zip(datas[2], pred_nums)])

    print(f"Accuracy: {accuracy * 100:.2f}%")
## 七、运行方法
1. 安装依赖

请先确保已安装 TensorFlow 2.x 及相关依赖：

pip install tensorflow numpy tqdm
2. 运行训练程序
python Learn2Carry-exercise.py
3. 使用 PowerShell 设置参数运行

本项目支持通过环境变量设置训练参数，减少频繁修改源码的操作。

$env:LEARN2CARRY_SEED=42
$env:LEARN2CARRY_TRAIN_STEPS=50
$env:LEARN2CARRY_TRAIN_BATCH=128
$env:LEARN2CARRY_EVAL_BATCH=200
$env:LEARN2CARRY_LOG_INTERVAL=10
$env:LEARN2CARRY_REPORT_OUT="outputs/learn2carry_report.json"

python .\Learn2Carry-exercise.py
## 八、工程化改进说明

在不改变 RNN 加法核心逻辑的前提下，对 Learn2Carry-exercise.py 进行了小幅工程化优化。

1. 新增环境变量配置

支持通过环境变量控制训练和评估参数：

环境变量	说明
LEARN2CARRY_SEED	随机种子
LEARN2CARRY_TRAIN_STEPS	训练步数
LEARN2CARRY_TRAIN_BATCH	训练 batch size
LEARN2CARRY_EVAL_BATCH	评估 batch size
LEARN2CARRY_MAXLEN	最大数位长度
LEARN2CARRY_LOG_INTERVAL	日志打印间隔
LEARN2CARRY_REPORT_OUT	实验报告输出路径
2. 增强实验复现性

程序中增加随机种子设置，固定 numpy 和 tensorflow 的随机状态，使实验结果更容易复现。

3. 增加结果留档

运行完成后，程序会自动保存实验报告，默认输出路径为：

outputs/learn2carry_report.json

报告中可记录训练参数、评估准确率等信息，方便后续查看和对比实验结果。

## 九、参数调整建议
参数	说明	建议
maxlen	最大数位长度	根据加法数字位数调整
batch_size	批量大小	增大可提升训练稳定性，但会增加内存占用
train_steps	训练步数	步数越多通常效果越好
RNN hidden size	RNN 隐藏层维度	可尝试从 64 调整为 128
learning_rate	学习率	可根据 loss 变化情况适当调整
## 十、实验结果示例

训练过程中可能输出如下日志：

>>>>>>> upstream/main
Step 0: Loss = 2.3026
Step 50: Loss = 0.9543
Step 100: Loss = 0.6821
...
Step 3000: Loss = 0.1235
Accuracy: 98.75%
<<<<<<< HEAD
```
=======

实际结果会受到训练步数、batch size、随机种子和运行环境影响。

## 十一、项目文件说明
.
├── Learn2Carry-exercise.py       # RNN 大数加法实验代码
├── rnn.py                        # PyTorch 版本 RNN 相关代码
├── outputs/                      # 实验结果输出目录
│   └── learn2carry_report.json   # 自动生成的实验报告
└── README.md                     # 项目说明文档
## 十二、总结

本项目通过唐诗生成和大数加法预测两个任务，展示了循环神经网络在序列建模任务中的应用。

其中，唐诗生成任务体现了 RNN 在自然语言生成中的作用；大数加法任务则展示了 RNN 对序列规律和进位逻辑的学习能力。通过本次工程化优化，代码在可配置性、可复现性和结果记录方面更加完善，便于后续实验、调试和扩展。
>>>>>>> upstream/main
