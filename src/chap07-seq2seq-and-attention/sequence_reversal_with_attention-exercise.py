#!/usr/bin/env python
# coding: utf-8

# # 序列逆置 （加注意力的seq2seq）
# 使用attentive sequence to sequence 模型将一个字符串序列逆置。例如 `OIMESIQFIQ` 逆置成 `QIFQISEMIO`(下图来自网络，是一个加attentino的sequence to sequence 模型示意图)
# ![attentive seq2seq](./seq2seq-attn.jpg)

# In[19]:


import numpy as np
import tensorflow as tf
import collections
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
import os,sys,tqdm


# ## 玩具序列数据生成
# 生成只包含[A-Z]的字符串，并且将encoder输入以及decoder输入以及decoder输出准备好（转成index）

# In[20]:


import random
import string

def randomString(stringLength):
    """生成一个指定长度的随机字符串，字符串由大写字母组成"""
    letters = string.ascii_uppercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def get_batch(batch_size, length):
    """
    生成一批用于训练的序列数据
    
    参数:
    batch_size: 批次大小
    length: 每个序列的长度
    
    返回:
    batched_examples: 原始字符串列表
    enc_x: 编码器输入，字符转换为索引后的张量
    dec_x: 解码器输入，包含起始标记，字符转换为索引后的张量
    y: 解码器目标输出，原始序列的逆序，字符转换为索引后的张量
    """
    # 生成随机字符串
    batched_examples = [randomString(length) for i in range(batch_size)]
    
    # 将字符串转换为索引 (A=1, B=2, ..., Z=26)
    enc_x = [[ord(ch)-ord('A')+1 for ch in list(exp)] for exp in batched_examples]
    
    # 目标输出是输入序列的逆序
    y = [[o for o in reversed(e_idx)] for e_idx in enc_x]
    
    # 解码器输入在目标序列前加一个起始标记(0)，并去掉最后一个字符
    dec_x = [[0]+e_idx[:-1] for e_idx in y]
    
    return (batched_examples, tf.constant(enc_x, dtype=tf.int32), 
            tf.constant(dec_x, dtype=tf.int32), tf.constant(y, dtype=tf.int32))

print(get_batch(2, 10))


# # 建立sequence to sequence 模型
# ##
# 完成两空，模型搭建以及单步解码逻辑

# In[26]:


class mySeq2SeqModel(keras.Model):
    def __init__(self):
        super(mySeq2SeqModel, self).__init__()
        
        # 词汇表大小: 26个字母 + 1个填充标记(0)
        self.v_sz = 27
        
        # 隐藏层大小
        self.hidden = 128
        
        # 嵌入层，将字符索引转换为向量表示
        self.embed_layer = tf.keras.layers.Embedding(self.v_sz, 64, 
                                                    batch_input_shape=[None, None])
        
        # 编码器和解码器的RNN单元
        self.encoder_cell = tf.keras.layers.SimpleRNNCell(self.hidden)
        self.decoder_cell = tf.keras.layers.SimpleRNNCell(self.hidden)
        
        # 编码器和解码器RNN
        # return_sequences=True: 返回所有时间步的输出
        # return_state=True: 返回最终状态
        self.encoder = tf.keras.layers.RNN(self.encoder_cell, 
                                           return_sequences=True, return_state=True)
        self.decoder = tf.keras.layers.RNN(self.decoder_cell, 
                                           return_sequences=True, return_state=True)
        
        # 注意力机制相关的全连接层
        self.dense_attn = tf.keras.layers.Dense(self.hidden)
        
        # 输出层，将解码器状态映射到词汇表大小的输出
        self.dense = tf.keras.layers.Dense(self.v_sz)
        
        
    @tf.function
    def call(self, enc_ids, dec_ids):
        '''
        完成带attention机制的 sequence2sequence 模型的搭建，模块已经在`__init__`函数中定义好，
        用双线性attention，或者自己改一下`__init__`函数做加性attention
        
        参数:
        enc_ids: 编码器输入序列 [batch_size, seq_len]
        dec_ids: 解码器输入序列 [batch_size, seq_len]
        
        返回:
        logits: 模型预测的logits [batch_size, seq_len, v_sz]
        '''
        # 1. 编码器处理输入序列
        enc_emb = self.embed_layer(enc_ids)  # [batch_size, seq_len, emb_dim]
        enc_out, enc_state = self.encoder(enc_emb)  # enc_out: [batch_size, seq_len, hidden]
        
        # 2. 解码器处理目标序列
        dec_emb = self.embed_layer(dec_ids)  # [batch_size, seq_len, emb_dim]
        dec_out, _ = self.decoder(dec_emb, initial_state=enc_state)  # [batch_size, seq_len, hidden]
        
        # 3. 注意力机制计算
        # 计算注意力权重
        # enc_out: [batch_size, enc_seq_len, hidden]
        # dec_out: [batch_size, dec_seq_len, hidden]
        # 为了计算注意力，我们需要将它们扩展为 [batch_size, dec_seq_len, enc_seq_len, hidden]
        dec_out_expanded = tf.expand_dims(dec_out, axis=2)  # [batch_size, dec_seq_len, 1, hidden]
        enc_out_expanded = tf.expand_dims(enc_out, axis=1)  # [batch_size, 1, enc_seq_len, hidden]
        
        # 双线性注意力: score = dec_out * W * enc_out
        # 这里简化为: score = dec_out * enc_out (通过点积实现)
        attn_scores = tf.matmul(dec_out_expanded, enc_out_expanded, transpose_b=True)  # [batch_size, dec_seq_len, 1, enc_seq_len]
        attn_scores = tf.squeeze(attn_scores, axis=2)  # [batch_size, dec_seq_len, enc_seq_len]
        
        # 应用softmax获取注意力权重
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)  # [batch_size, dec_seq_len, enc_seq_len]
        
        # 计算上下文向量
        context = tf.matmul(attn_weights, enc_out)  # [batch_size, dec_seq_len, hidden]
        
        # 4. 结合上下文向量和解码器输出
        dec_out_with_context = tf.concat([dec_out, context], axis=-1)  # [batch_size, dec_seq_len, hidden*2]
        
        # 5. 通过全连接层生成最终输出
        logits = self.dense(dec_out_with_context)  # [batch_size, dec_seq_len, v_sz]
        
        return logits
    
    
    @tf.function
    def encode(self, enc_ids):
        """
        对输入序列进行编码
        
        参数:
        enc_ids: 编码器输入序列 [batch_size, seq_len]
        
        返回:
        enc_out: 编码器各时间步的输出 [batch_size, seq_len, hidden]
        state: 编码器最终状态 [batch_size, hidden]
        """
        # 将输入的token ID转换为密集向量表示
        enc_emb = self.embed_layer(enc_ids)  # shape(b_sz, len, emb_sz)
        enc_out, enc_state = self.encoder(enc_emb)
        return enc_out, [enc_out[:, -1, :], enc_state]
    
    def get_next_token(self, x, state, enc_out):
        '''
        单步解码逻辑，根据当前输入和状态预测下一个token
        
        参数:
        x: 当前输入token [batch_size,]
        state: 当前状态 [batch_size, hidden]
        enc_out: 编码器输出 [batch_size, enc_seq_len, hidden]
        
        返回:
        out: 预测的下一个token索引 [batch_size,]
        new_state: 新的状态 [batch_size, hidden]
        '''
        # 1. 嵌入当前输入token
        x_emb = self.embed_layer(x)  # [batch_size, emb_dim]
        
        # 2. 解码器单步更新
        dec_out, new_state = self.decoder_cell(x_emb, states=state)  # [batch_size, hidden]
        
        # 3. 注意力计算
        # 计算当前解码器状态与编码器各时间步输出的相似度
        dec_out_expanded = tf.expand_dims(dec_out, axis=1)  # [batch_size, 1, hidden]
        attn_scores = tf.matmul(dec_out_expanded, enc_out, transpose_b=True)  # [batch_size, 1, enc_seq_len]
        attn_scores = tf.squeeze(attn_scores, axis=1)  # [batch_size, enc_seq_len]
        
        # 应用softmax获取注意力权重
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)  # [batch_size, enc_seq_len]
        
        # 计算上下文向量
        context = tf.matmul(attn_weights, enc_out)  # [batch_size, hidden]
        
        # 4. 结合上下文向量和解码器输出
        dec_out_with_context = tf.concat([dec_out, context], axis=-1)  # [batch_size, hidden*2]
        
        # 5. 通过全连接层生成词汇表上的分布
        logits = self.dense(dec_out_with_context)  # [batch_size, v_sz]
        
        # 6. 获取概率最高的token索引
        out = tf.argmax(logits, axis=-1)  # [batch_size,]
        
        return out, new_state


# # Loss函数以及训练逻辑
# ##
# In[27]:


@tf.function
def compute_loss(logits, labels):
    """计算模型预测的损失"""
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
    losses = tf.reduce_mean(losses)
    return losses

@tf.function
def train_one_step(model, optimizer, enc_x, dec_x, y):
    """
    执行一步训练
    
    参数:
    model: 模型实例
    optimizer: 优化器
    enc_x: 编码器输入
    dec_x: 解码器输入
    y: 目标输出
    
    返回:
    loss: 计算的损失值
    """
    with tf.GradientTape() as tape:
        logits = model(enc_x, dec_x)
        loss = compute_loss(logits, y)

    # 计算梯度
    grads = tape.gradient(loss, model.trainable_variables)
    
    # 应用梯度
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    return loss

def train(model, optimizer, seqlen):
    """
    训练模型
    
    参数:
    model: 模型实例
    optimizer: 优化器
    seqlen: 序列长度
    
    返回:
    loss: 最终损失值
    """
    loss = 0.0
    accuracy = 0.0
    
    # 训练2000步
    for step in range(2000):
        batched_examples, enc_x, dec_x, y = get_batch(32, seqlen)
        loss = train_one_step(model, optimizer, enc_x, dec_x, y)
        
        # 每500步打印一次损失
        if step % 500 == 0:
            print('step', step, ': loss', loss.numpy())
    
    return loss


# # 训练迭代

# In[28]:


optimizer = optimizers.Adam(0.0005)
model = mySeq2SeqModel()
train(model, optimizer, seqlen=20)


# # 测试模型逆置能力
# 首先要先对输入的一个字符串进行encode，然后在用decoder解码出逆置的字符串
# 
# 测试阶段跟训练阶段的区别在于，在训练的时候decoder的输入是给定的，而在预测的时候我们需要一步步生成下一步的decoder的输入

# In[30]:


def sequence_reversal():
    """测试模型的序列逆置能力"""
    
    def decode(init_state, steps, enc_out):
        """
        自回归解码过程，从初始状态开始，一步步生成输出序列
        
        参数:
        init_state: 初始状态
        steps: 要生成的步数
        enc_out: 编码器输出
        
        返回:
        out: 生成的字符串列表
        """
        b_sz = tf.shape(init_state[0])[0]
        
        # 初始输入为0(起始标记)
        cur_token = tf.zeros(shape=[b_sz], dtype=tf.int32)
        state = init_state
        collect = []
        
        # 逐步生成输出序列
        for i in range(steps):
            cur_token, state = model.get_next_token(cur_token, state, enc_out)
            collect.append(tf.expand_dims(cur_token, axis=-1))
        
        # 将生成的索引转换为字符串
        out = tf.concat(collect, axis=-1).numpy()
        out = [''.join([chr(idx+ord('A')-1) for idx in exp]) for exp in out]
        
        return out
    
    # 生成一批测试数据
    batched_examples, enc_x, _, _ = get_batch(32, 20)
    
    # 编码输入序列
    enc_out, state = model.encode(enc_x)
    
    # 解码生成逆置序列
    return decode(state, enc_x.get_shape()[-1], enc_out), batched_examples

def is_reverse(seq, rev_seq):
    """检查一个序列是否是另一个序列的逆序"""
    rev_seq_rev = ''.join([i for i in reversed(list(rev_seq))])
    if seq == rev_seq_rev:
        return True
    else:
        return False
# 测试函数功能
print([is_reverse(*item) for item in list(zip(*sequence_reversal()))])
print(list(zip(*sequence_reversal())))


# In[ ]:

