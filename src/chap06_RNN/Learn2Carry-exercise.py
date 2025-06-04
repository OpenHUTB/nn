#!/usr/bin/env python
# coding: utf-8
# # 加法进位实验
# <img src="https://github.com/JerrikEph/jerrikeph.github.io/raw/master/Learn2Carry.png" width=650>

# In[1]:
#导入了多个用于构建和训练深度学习模型的Python库和模块
import numpy as np
import tensorflow as tf
import collections
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers, optimizers, datasets
import os,sys,tqdm


# ## 数据生成
# 我们随机在 `start->end`之间采样除整数对`(num1, num2)`，计算结果`num1+num2`作为监督信号。
# * 首先将数字转换成数字位列表 `convertNum2Digits`
# * 将数字位列表反向
# * 将数字位列表填充到同样的长度 `pad2len`

# In[2]:

def gen_data_batch(batch_size, start, end):
    '''在(start, end)区间采样生成一个batch的整型的数据
    Args :
        batch_size: batch_size
        start: 开始数值
        end: 结束数值
    '''
    numbers_1 = np.random.randint(start, end, batch_size)
    numbers_2 = np.random.randint(start, end, batch_size)
    results = numbers_1 + numbers_2
    return numbers_1, numbers_2, results

def convertNum2Digits(Num):
    '''将一个整数转换成一个数字位的列表,例如 133412 ==> [1, 3, 3, 4, 1, 2]
    '''
    strNum = str(Num)
    chNums = list(strNum)
    digitNums = [int(o) for o in strNum]
    return digitNums

def convertDigits2Num(Digits):
    '''将数字位列表反向， 例如 [1, 3, 3, 4, 1, 2] ==> [2, 1, 4, 3, 3, 1]
    '''
    digitStrs = [str(o) for o in Digits]
    numStr = ''.join(digitStrs)
    Num = int(numStr)
    return Num

def pad2len(lst, length, pad=0):
    '''将一个列表用`pad`填充到`length`的长度 例如 pad2len([1, 3, 2, 3], 6, pad=0) ==> [1, 3, 2, 3, 0, 0]
    '''
    lst+=[pad]*(length - len(lst))
    return lst

def results_converter(res_lst):
    '''将预测好的数字位列表批量转换成为原始整数
    Args:
        res_lst: shape(b_sz, len(digits))
    '''
    res = [reversed(digits) for digits in res_lst]
    return [convertDigits2Num(digits) for digits in res]

def prepare_batch(Nums1, Nums2, results, maxlen):
    '''准备一个batch的数据，将数值转换成反转的数位列表并且填充到固定长度
    Args:
        Nums1: shape(batch_size,)
        Nums2: shape(batch_size,)
        results: shape(batch_size,)
        maxlen:  type(int)
    Returns:
        Nums1: shape(batch_size, maxlen)
        Nums2: shape(batch_size, maxlen)
        results: shape(batch_size, maxlen)
    '''
    Nums1 = [convertNum2Digits(o) for o in Nums1]
    Nums2 = [convertNum2Digits(o) for o in Nums2]
    results = [convertNum2Digits(o) for o in results]
    
    Nums1 = [list(reversed(o)) for o in Nums1]
    Nums2 = [list(reversed(o)) for o in Nums2]
    results = [list(reversed(o)) for o in results]
    
    Nums1 = [pad2len(o, maxlen) for o in Nums1]
    Nums2 = [pad2len(o, maxlen) for o in Nums2]
    results = [pad2len(o, maxlen) for o in results]
    
    return Nums1, Nums2, results

# # 建模过程， 按照图示完成建模

# In[3]:

class myRNNModel(keras.Model):
    def __init__(self):
        super(myRNNModel, self).__init__()
        self.embed_layer = tf.keras.layers.Embedding(10, 32, 
                                                    batch_input_shape=[None, None])
        
        self.rnncell = tf.keras.layers.SimpleRNNCell(64)
        self.rnn_layer = tf.keras.layers.RNN(self.rnncell, return_sequences=True)
        self.dense = tf.keras.layers.Dense(10)
        
    @tf.function
    def call(self, num1, num2):
        '''
        此处完成上述图中模型
        '''
         # 输入形状: (batch_size, maxlen)
        embedded_num1 = self.embed_layer(num1)  # (batch_size, maxlen, 32)
        embedded_num2 = self.embed_layer(num2)  # (batch_size, maxlen, 32)
        
        # 特征融合：将两个数字的嵌入表示相加
        combined_features = tf.add(embedded_num1, embedded_num2)
        
        # RNN处理序列依赖
        rnn_outputs = self.rnn_layer(combined_features)  # (batch_size, maxlen, 64)
        
        # 全连接层输出每个位置的数字预测（0-9）
        logits = self.dense(rnn_outputs)  # (batch_size, maxlen, 10)
        return logits
        
# In[4]:
@tf.function
def compute_loss(logits, labels):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
    return tf.reduce_mean(losses)## 使用交叉熵损失函数来计算每个样本的损失

@tf.function
def train_one_step(model, optimizer, x, y, label):
    with tf.GradientTape() as tape:
        logits = model(x, y)       #前向传播：计算输出 logits
        loss = compute_loss(logits, label)

    # compute gradient
    grads = tape.gradient(loss, model.trainable_variables)#计算梯度，自动求导
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def train(steps, model, optimizer):
    loss = 0.0
    accuracy = 0.0
    # 训练循环，执行指定步数
for step in range(steps):
    # 生成批量数据（200个样本），数值范围0-555555555
    datas = gen_data_batch(batch_size=200, start=0, end=555555555)
    
    # 准备训练数据：两个数字序列和结果，统一填充到长度11
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=11)
    
    # 单步训练：将数据转为Tensor，计算并返回损失
    loss = train_one_step(model, optimizer, 
                          tf.constant(Nums1, dtype=tf.int32),
                          tf.constant(Nums2, dtype=tf.int32),
                          tf.constant(results, dtype=tf.int32))
    
    # 每50步打印当前损失
    if step % 50 == 0:
        print('step', step, ': loss', loss.numpy())

return loss  # 返回最终损失值

def evaluate(model):
    datas = gen_data_batch(batch_size=2000, start=555555555, end=999999999)
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=11)
    # 前向传播预测 logits
    logits = model(tf.constant(Nums1, dtype=tf.int32), tf.constant(Nums2, dtype=tf.int32))
    logits = logits.numpy()
    pred = np.argmax(logits, axis=-1) # 每位取最大概率的数字
    res = results_converter(pred)
    for o in list(zip(datas[2], res))[:20]:
        print(o[0], o[1], o[0]==o[1])

    print('accuracy is: %g' % np.mean([o[0]==o[1] for o in zip(datas[2], res)]))


# In[5]:

optimizer = optimizers.Adam(0.001)
model = myRNNModel()


# In[6]:

train(3000, model, optimizer)
evaluate(model)


# In[7]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




