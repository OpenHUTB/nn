# !/usr/bin/env python
# coding: utf-8

# # 序列逆置 （加注意力的seq2seq）
# 使用attentive sequence to sequence 模型将一个字符串序列逆置。例如 `OIMESIQFIQ` 逆置成 `QIFQISEMIO`(下图来自网络，是一个加attentino的sequence to sequence 模型示意图)
# ![attentive seq2seq](./seq2seq-attn.jpg)

# In[19]:


# 导入NumPy库，用于科学计算和数组操作
import numpy as np
# 导入TensorFlow库，用于机器学习和深度学习任务
import tensorflow as tf
# 导入 collections 模块，提供额外的容器类型如字典和列表
import collections
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
import os,sys,tqdm


# ## 玩具序列数据生成
# 生成只包含[A-Z]的字符串，并且将encoder输入以及decoder输入以及decoder输出准备好（转成index）

# In[20]:

# 导入random模块，用于生成随机数和随机操作
import random
# 导入string模块，提供与字符串操作相关的常量和工具函数
import string

def randomString(stringLength):  # 定义函数 get_batch，输入参数 batch_size 和 length
    """Generate a random string with the combination of lowercase and uppercase letters """


    letters = string.ascii_uppercase #生成指定长度的随机大写字母字符串
    return ''.join(random.choice(letters) for i in range(stringLength))

def get_batch(batch_size, length):
    batched_examples = [randomString(length) for i in range(batch_size)]   # 生成 batch_size 个长度为 length 的随机字符串
    enc_x = [[ord(ch)-ord('A')+1 for ch in list(exp)] for exp in batched_examples]   # 将每个字符串中的字符转换为对应的数字编码
    y = [[o for o in reversed(e_idx)] for e_idx in enc_x]  # 生成目标输出 y，将 enc_x 中的每个编码列表反转
    dec_x = [[0]+e_idx[:-1] for e_idx in y]   # 生成解码输入 dec_x，将 y 中的每个编码列表左移一位，并在开头添加 0
    return (batched_examples, tf.constant(enc_x, dtype=tf.int32),    # 返回原始字符串列表、编码后的输入、编码后的解码输入和编码后的目标输出
            tf.constant(dec_x, dtype=tf.int32), tf.constant(y, dtype=tf.int32))  # 使用 tf.constant 将列表转换为 TensorFlow 张量
print(get_batch(2, 10))


# # 建立sequence to sequence 模型
# 完成两空，模型搭建以及单步解码逻辑

# In[26]:

# 定义了一个名为 mySeq2SeqModel 的类，继承自 keras.Model
# 调用父类 keras.Model 的初始化方法
class mySeq2SeqModel(keras.Model):
    def __init__(self):
        super(mySeq2SeqModel, self).__init__()
         # 模型参数配置
        self.v_sz=27 # 词汇表大小（包括可能的特殊符号）
        self.hidden = 128 # 隐藏层维度/RNN单元的大小
        self.embed_layer = tf.keras.layers.Embedding(self.v_sz, 64, 
                                                    batch_input_shape=[None, None]) 
        # 输入词汇表大小，即嵌入层的输入维度和每个词向量的维度以及输入张量的形状，支持任意批次大小和序列长度
        
        self.encoder_cell = tf.keras.layers.SimpleRNNCell(self.hidden) 
        # 隐藏状态的维度，即输出维度
        self.decoder_cell = tf.keras.layers.SimpleRNNCell(self.hidden)
        
        self.encoder = tf.keras.layers.RNN(self.encoder_cell, 
                                           return_sequences=True, return_state=True)
        self.decoder = tf.keras.layers.RNN(self.decoder_cell, 
                                           return_sequences=True, return_state=True)
        # 注意力机制和输出层
        self.dense_attn = tf.keras.layers.Dense(self.hidden)
        self.dense = tf.keras.layers.Dense(self.v_sz) # 输出层：将解码器状态映射到词汇表空间（预测下一个词）
        
        
    @tf.function
    def call(self, enc_ids, dec_ids):
        '''
        todo
        
        完成带attention机制的 sequence2sequence 模型的搭建，模块已经在`__init__`函数中定义好，
        用双线性attention，或者自己改一下`__init__`函数做加性attention
        '''
        # Embedding
        enc_emb = self.embed_layer(enc_ids)  # (B, T1, E)
        dec_emb = self.embed_layer(dec_ids)  # (B, T2, E)

        # 编码器输出
        enc_outputs, enc_state = self.encoder(enc_emb)  # (B, T1, H), (B, H)

        # 初始化解码器状态
        state = enc_state  # (B, H)

        # Attention + Decoder + Output
        #这里的这段代码实现了一个加性注意力机制
        outputs = []
        for t in range(dec_emb.shape[1]):
            # 当前时刻的 decoder 输入
            dec_input_t = dec_emb[:, t, :]  # (B, E)

            # Attention 权重计算（加性）
            score = tf.nn.tanh(self.dense_attn(enc_outputs))  # (B, T1, H)
            score = tf.reduce_sum(score * tf.expand_dims(state, 1), axis=-1)  # (B, T1)
            attn_weights = tf.nn.softmax(score, axis=-1)  # (B, T1)
            context = tf.reduce_sum(enc_outputs * tf.expand_dims(attn_weights, -1), axis=1)  # (B, H)

            # 合并 context 与 decoder input
            dec_input_combined = tf.concat([dec_input_t, context], axis=-1)  # (B, H+E)

            # Decoder RNN 一步
            output, state = self.decoder_cell(dec_input_combined, [state])  # output: (B, H)

            # 输出层
            logits = self.dense(output)  # (B, V)
            outputs.append(tf.expand_dims(logits, axis=1))  # (B, 1, V)

        logits = tf.concat(outputs, axis=1)  # (B, T2, V)
        return logits
    
    
    @tf.function
    #这段代码定义了一个编码器（encode）方法，主要功能是将输入的 token 序列（enc_ids）转换为带有上下文信息的表示（enc_out）和最终状态（enc_state）
    def encode(self, enc_ids):
        enc_emb = self.embed_layer(enc_ids) # shape(b_sz, len, emb_sz)
        enc_out, enc_state = self.encoder(enc_emb)
        return enc_out, [enc_out[:, -1, :], enc_state]
    
    def get_next_token(self, x, state, enc_out):
        '''
        shape(x) = [b_sz,] 
        '''
    
        '''
        todo
        参考sequence_reversal-exercise, 自己构建单步解码逻辑
        '''
        x_embed = self.embed_layer(x)  # (B, E)

        # Attention 权重计算
        score = tf.nn.tanh(self.dense_attn(enc_out))  # (B, T1, H)
        score = tf.reduce_sum(score * tf.expand_dims(state, 1), axis=-1)  # (B, T1)
        attn_weights = tf.nn.softmax(score, axis=-1)  # (B, T1)
        context = tf.reduce_sum(enc_out * tf.expand_dims(attn_weights, -1), axis=1)  # (B, H)

        # 合并 decoder input 和 context
        rnn_input = tf.concat([x_embed, context], axis=-1)  # (B, H + E)

        # 解码一步
        output, state = self.decoder_cell(rnn_input, [state])  # output: (B, H)
        logits = self.dense(output)  #全连接层 self.dense 将 RNN 输出映射到词表空间
        next_token = tf.argmax(logits, axis=-1, output_type=tf.int32)  # (B,)
        return next_token, state[0]
        


# # Loss函数以及训练逻辑
# In[27]:


@tf.function
#这段代码定义了一个计算分类任务损失值的函数 compute_loss，主要用于神经网络训练时评估预测结果与真实标签之间的差异。
def compute_loss(logits, labels):
    # 计算每个位置上的交叉熵损失
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
    # 对整个批次求平均
    losses = tf.reduce_mean(losses)
    return losses
    
# 使用图执行模式优化训练步骤
@tf.function
def train_one_step(model, optimizer, enc_x, dec_x, y):
     # 创建梯度记录环境
    with tf.GradientTape() as tape:
          # 前向传播：获取模型预测
        logits = model(enc_x, dec_x)
        # 计算预测结果与真实标签之间的损失
        loss = compute_loss(logits, y)

    # compute gradient
    # 计算损失对可训练参数的梯度
    grads = tape.gradient(loss, model.trainable_variables)
    # 应用梯度更新模型参数
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def train(model, optimizer, seqlen):
    loss = 0.0
    accuracy = 0.0
        # 循环2000个训练步骤
    for step in range(2000):
        # 获取一个批次的数据：
        # batched_examples: 原始字符串样本列表
        # enc_x: 编码器输入序列
        # dec_x: 解码器输入序列
        # y: 目标输出序列
        batched_examples, enc_x, dec_x, y = get_batch(32, seqlen)
        # 执行训练步骤
        loss = train_one_step(model, optimizer, enc_x, dec_x, y)
        # 每500步打印一次进度
        if step % 500 == 0:
            print('step', step, ': loss', loss.numpy())
    return loss


# # 训练迭代
# In[28]:


# 创建Adam优化器实例，设置学习率为0.0005
optimizer = optimizers.Adam(0.0005)
# 实例化自定义的序列到序列模型
model = mySeq2SeqModel()
# 调用训练函数，传入模型、优化器和序列长度参数
train(model, optimizer, seqlen=20)


# # 测试模型逆置能力
# 首先要先对输入的一个字符串进行encode，然后在用decoder解码出逆置的字符串
# 测试阶段跟训练阶段的区别在于，在训练的时候decoder的输入是给定的，而在预测的时候我们需要一步步生成下一步的decoder的输入
# In[30]:


def sequence_reversal():
    def decode(init_state, steps, enc_out):
        # 获取批次大小
        b_sz = tf.shape(init_state[0])[0]
        # 初始化解码器输入：全0（表示起始符）
        cur_token = tf.zeros(shape=[b_sz], dtype=tf.int32)
        state = init_state
        collect = [] # 收集每一步生成的token
        # 逐步生成序列
        for i in range(steps):
             # 获取下一个token和新的状态
            cur_token, state = model.get_next_token(cur_token, state, enc_out)
            # 收集当前步生成的token
            collect.append(tf.expand_dims(cur_token, axis=-1))
            # 拼接所有生成的token [batch_size, steps]
        out = tf.concat(collect, axis=-1).numpy()
        # 将索引转换为字符：
        # 假设索引1对应'A'，2对应'B'，依此类推  
        out = [''.join([chr(idx+ord('A')-1) for idx in exp]) for exp in out]
        return out
     # 获取测试批次（仅使用编码器输入）
    batched_examples, enc_x, _, _ = get_batch(32, 20)
    # 编码输入序列
    enc_out, state = model.encode(enc_x)
    # 解码生成逆置序列
    return decode(state, enc_x.get_shape()[-1], enc_out), batched_examples

def is_reverse(seq, rev_seq):# 将待检测序列反转（使用列表推导式+reversed函数实现字符串反转）
    # 比较原始序列与反转后的序列是否相同
    rev_seq_rev = ''.join([i for i in reversed(list(rev_seq))])
    if seq == rev_seq_rev:
        return True
    else:
        return False

print([is_reverse(*item) for item in list(zip(*sequence_reversal()))])
print(list(zip(*sequence_reversal())))


# In[ ]:




