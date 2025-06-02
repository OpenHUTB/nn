#!/usr/bin/env python
# coding: utf-8

# # 序列逆置
# 使用sequence to sequence 模型将一个字符串序列逆置。
# 例如 `OIMESIQFIQ` 逆置成 `QIFQISEMIO`(下图来自网络，是一个sequence to sequence 模型示意图 )
# ![seq2seq](./seq2seq.png)

# In[1]:

import numpy as np
import tensorflow as tf
import collections
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers, optimizers, datasets
import os,sys,tqdm


# ## 玩具序列数据生成
# 生成只包含[A-Z]的字符串，并且将encoder输入以及decoder输入以及decoder输出准备好（转成index）

# In[2]:

import random
import string

def randomString(stringLength):
    """Generate a random string with the combination of lowercase and uppercase letters """

    letters = string.ascii_uppercase # 定义可用的字符集
    return ''.join(random.choice(letters) for i in range(stringLength))
    # 生成随机字符串
    # 使用 random.choice(letters) 从 letters 中随机选择一个字符
    # 重复这个过程 stringLength 次，并用 ''.join() 将这些字符连接成一个字符串
    # 最终返回生成的随机字符串

def get_batch(batch_size, length):
    batched_examples = [randomString(length) for i in range(batch_size)]
    enc_x = [[ord(ch) - ord('A') + 1 for ch in list(exp)] for exp in batched_examples]
    y = [[o for o in reversed(e_idx)] for e_idx in enc_x]
    dec_x = [[0] + e_idx[:-1] for e_idx in y]
    return (batched_examples, tf.constant(enc_x, dtype=tf.int32), 
            tf.constant(dec_x, dtype=tf.int32), tf.constant(y, dtype=tf.int32))
print(get_batch(2, 10))


# # 建立sequence to sequence 模型

# In[3]:


class mySeq2SeqModel(keras.Model):
    def __init__(self):
        # 初始化父类 keras.Model，必须调用
        super(mySeq2SeqModel, self).__init__()

        # 词表大小为27：A-Z共26个大写字母，加上1个特殊的起始符（用0表示）
        self.v_sz = 27

        # 嵌入层：将每个字符的索引映射成64维的向量表示
        # 输入维度：self.v_sz（即词表大小），输出维度为64
        self.embed_layer = tf.keras.layers.Embedding(self.v_sz, 64,
                                                    batch_input_shape=[None, None])

        # 编码器RNN单元：使用SimpleRNNCell，隐藏状态维度为128
        self.encoder_cell = tf.keras.layers.SimpleRNNCell(128)

        # 解码器RNN单元：使用SimpleRNNCell，隐藏状态维度为128
        self.decoder_cell = tf.keras.layers.SimpleRNNCell(128)

        # 编码器RNN层：将RNNCell包裹成完整RNN，输出整个序列（return_sequences=True），并返回最终状态（return_state=True）
        self.encoder = tf.keras.layers.RNN(
            self.encoder_cell,
            return_sequences=True,   # 返回每个时间步的输出
            return_state=True        # 还返回最终隐藏状态
        )

        # 解码器RNN层：与编码器类似
        self.decoder = tf.keras.layers.RNN(
            self.decoder_cell,
            return_sequences=True,
            return_state=True
        )

        # 全连接层：将解码器的每个时间步的输出转换为词表大小的 logits（即每个字符的预测概率分布）
        self.dense = tf.keras.layers.Dense(self.v_sz)

        
    @tf.function
    def call(self, enc_ids, dec_ids):
        '''
        完成sequence2sequence 模型的搭建，模块已经在`__init__`函数中定义好
        '''
        # 编码过程
        enc_emb = self.embed_layer(enc_ids)  # (batch_size, enc_seq_len, emb_dim)
        enc_out, enc_state = self.encoder(enc_emb)  # enc_out: (batch_size, enc_seq_len, enc_units)
        
        # 解码过程，使用编码器的最终状态作为初始状态
        dec_emb = self.embed_layer(dec_ids)  # (batch_size, dec_seq_len, emb_dim)
        dec_out, dec_state = self.decoder(dec_emb, initial_state=enc_state)  # dec_out: (batch_size, dec_seq_len, dec_units)
        
        # 计算logits
        logits = self.dense(dec_out)  # (batch_size, dec_seq_len, vocab_size)
        return logits
    
    
    @tf.function
    def encode(self, enc_ids):
        enc_emb = self.embed_layer(enc_ids) # shape(b_sz, len, emb_sz)
        enc_out, enc_state = self.encoder(enc_emb)
        
        return [enc_out[:, -1, :], enc_state]
    
    def get_next_token(self, x, state):
        '''
        shape(x) = [b_sz,] 
        '''
        inp_emb = self.embed_layer(x) #shape(b_sz, emb_sz)
        h, state = self.decoder_cell.call(inp_emb, state) # shape(b_sz, h_sz)
        logits = self.dense(h) # shape(b_sz, v_sz)
        out = tf.argmax(logits, axis=-1)
        return out, state


# # Loss函数以及训练逻辑

# In[4]:

#定义了一个使用TensorFlow的@tf.function装饰器的函数compute_loss，用于计算模型预测的损失值
@tf.function
def compute_loss(logits, labels):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
    losses = tf.reduce_mean(losses)
    return losses
#定义了一个使用TensorFlow的@tf.function装饰器的函数train_one_step，用于执行一个训练步骤
@tf.function
def train_one_step(model, optimizer, enc_x, dec_x, y):
    with tf.GradientTape() as tape:
        logits = model(enc_x, dec_x)
        loss = compute_loss(logits, y)

    # compute gradient
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def train(model, optimizer, seqlen):
    loss = 0.0
    accuracy = 0.0
    for step in range(3000):
        batched_examples, enc_x, dec_x, y = get_batch(32, seqlen)
        loss = train_one_step(model, optimizer, enc_x, dec_x, y)
        if step % 500 == 0:
            print('step', step, ': loss', loss.numpy())
    return loss


# # 训练迭代

# In[5]:
optimizer = optimizers.Adam(0.0005)
model = mySeq2SeqModel()
train(model, optimizer, seqlen=20)


# # 测试模型逆置能力
# 首先要先对输入的一个字符串进行encode，然后在用decoder解码出逆置的字符串
# 测试阶段跟训练阶段的区别在于，在训练的时候decoder的输入是给定的，而在预测的时候我们需要一步步生成下一步的decoder的输入

# In[6]:


def sequence_reversal():
    def decode(init_state, steps=10):
        b_sz = tf.shape(init_state[0])[0]
        cur_token = tf.zeros(shape=[b_sz], dtype=tf.int32)
        state = init_state
        collect = []
        for i in range(steps):
            cur_token, state = model.get_next_token(cur_token, state)
            collect.append(tf.expand_dims(cur_token, axis=-1))
        out = tf.concat(collect, axis=-1).numpy()
        out = [''.join([chr(idx+ord('A')-1) for idx in exp]) for exp in out]
        return out
    
    batched_examples, enc_x, _, _ = get_batch(32, 10)
    state = model.encode(enc_x)
    return decode(state, enc_x.get_shape()[-1]), batched_examples

def is_reverse(seq, rev_seq):
    rev_seq_rev = ''.join([i for i in reversed(list(rev_seq))])
    if seq == rev_seq_rev:
        return True
    else:
        return False
print([is_reverse(*item) for item in list(zip(*sequence_reversal()))])
print(list(zip(*sequence_reversal())))


# In[ ]:




