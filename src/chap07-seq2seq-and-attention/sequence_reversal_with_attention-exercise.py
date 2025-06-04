#!/usr/bin/env python
# coding: utf-8

# 序列逆置 （加注意力的seq2seq）
# 使用attentive sequence to sequence 模型将一个字符串序列逆置。例如 `OIMESIQFIQ` 逆置成 `QIFQISEMIO`

import numpy as np
import tensorflow as tf
import collections
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers, optimizers, datasets
import os, sys, tqdm
import random
import string

# 玩具序列数据生成
# 生成只包含[A-Z]的字符串，并且将encoder输入以及decoder输入以及decoder输出准备好（转成index）

def randomString(stringLength):
    """生成指定长度的随机字符串（仅含大写字母A-Z）"""
    letters = string.ascii_uppercase
    return ''.join(random.choice(letters) for i in range(stringLength))

 #定义一个函数 get_batch，用于生成一批训练数据
def get_batch(batch_size, length):
    """生成一批训练数据
    返回:
        batched_examples: 原始字符串列表
        enc_x: 编码器输入（字符转索引）
        dec_x: 解码器输入（带起始标记的逆置序列）
        y: 解码器目标输出（逆置序列）
    """
    batched_examples = [randomString(length) for i in range(batch_size)]
    
    # 将字符转换为索引（A=1, B=2, ..., Z=26）
    enc_x = [[ord(ch)-ord('A')+1 for ch in list(exp)] for exp in batched_examples]
    
    # 目标输出：将输入序列逆置
    y = [[o for o in reversed(e_idx)] for e_idx in enc_x]
    
    # 解码器输入：在目标序列前加0（起始标记），并去掉最后一个字符
    dec_x = [[0]+e_idx[:-1] for e_idx in y]
    
    return (batched_examples, tf.constant(enc_x, dtype=tf.int32), 
            tf.constant(dec_x, dtype=tf.int32), tf.constant(y, dtype=tf.int32))

print(get_batch(2, 10))

# # 建立sequence to sequence 模型
# 完成两空，模型搭建以及单步解码逻辑

# 定义了一个自定义的序列到序列（Seq2Seq）模型类mySeq2SeqModel，继承自keras.Model
class mySeq2SeqModel(keras.Model):
    def __init__(self):
        super(mySeq2SeqModel, self).__init__()
        self.v_sz = 27  # 词汇表大小：A-Z(26) + 填充符0(1)
        self.hidden = 128  # 隐藏层维度
        
        # 嵌入层：将字符索引转换为向量
        self.embed_layer = tf.keras.layers.Embedding(self.v_sz, 64, 
                                                    batch_input_shape=[None, None])
        
        # 编码器和解码器的RNN单元
        self.encoder_cell = tf.keras.layers.SimpleRNNCell(self.hidden)
        self.decoder_cell = tf.keras.layers.SimpleRNNCell(self.hidden)
        
        # 编码器：处理输入序列，返回所有时间步的隐藏状态和最终状态
        self.encoder = tf.keras.layers.RNN(self.encoder_cell, 
                                           return_sequences=True, return_state=True)
        # 解码器：生成输出序列
        self.decoder = tf.keras.layers.RNN(self.decoder_cell, 
                                           return_sequences=True, return_state=True)
        
        # 注意力机制相关层
        self.dense_attn = tf.keras.layers.Dense(self.hidden)
        # 输出层：将隐藏状态映射到词汇表大小的向量
        self.dense = tf.keras.layers.Dense(self.v_sz)
        
        
    @tf.function
    def call(self, enc_ids, dec_ids):
        '''
        实现带attention机制的 sequence2sequence 模型
        使用双线性attention计算解码器对编码器各位置的关注度
        
        参数:
            enc_ids: 编码器输入序列 [batch_size, enc_len]
            dec_ids: 解码器输入序列 [batch_size, dec_len]
        返回:
            logits: 预测结果 [batch_size, dec_len, v_sz]
        '''
        # 编码器处理输入序列
        enc_emb = self.embed_layer(enc_ids)  # [batch_size, enc_len, emb_dim]
        enc_outputs, enc_state = self.encoder(enc_emb)  # enc_outputs: [batch_size, enc_len, hidden]
        
        # 解码器处理目标序列（训练时使用Teacher Forcing）
        dec_emb = self.embed_layer(dec_ids)  # [batch_size, dec_len, emb_dim]
        dec_outputs, _ = self.decoder(dec_emb, initial_state=enc_state)  # [batch_size, dec_len, hidden]
        
        # 注意力机制：计算解码器对编码器各位置的关注度
        context = tf.keras.layers.Attention()([dec_outputs, enc_outputs])
        
        # 合并解码器输出和上下文向量
        combined = tf.concat([dec_outputs, context], axis=-1)  # [batch_size, dec_len, 2*hidden]
        
        # 生成预测logits
        attn_output = self.dense_attn(combined)  # [batch_size, dec_len, hidden]
        logits = self.dense(attn_output)  # [batch_size, dec_len, v_sz]
        return logits
       
    
    @tf.function
    def encode(self, enc_ids):
        """对输入序列进行编码，返回编码器输出和初始状态"""
        enc_emb = self.embed_layer(enc_ids)  # [batch_size, len, emb_dim]
        enc_out, enc_state = self.encoder(enc_emb)  # enc_out: [batch_size, len, hidden]
        return enc_out, [enc_out[:, -1, :], enc_state]  # 返回编码器输出和最终状态
    
    def get_next_token(self, x, state, enc_out):
        '''
        单步解码逻辑：根据当前输入和状态预测下一个token
        shape(x) = [batch_size,] 
        '''
        # 嵌入当前输入token
        emb_x = self.embed_layer(x)  # [batch_size, emb_dim]
        
        # RNN单步计算，更新状态
        output, new_state = self.decoder_cell(emb_x, state)  # output: [batch_size, hidden]
        
        # 注意力计算：基于当前解码器状态关注编码器的不同位置
        output_expanded = tf.expand_dims(output, 1)  # [batch_size, 1, hidden]
        scores = tf.matmul(output_expanded, enc_out, transpose_b=True)  # [batch_size, 1, enc_len]
        scores = tf.squeeze(scores, axis=1)  # [batch_size, enc_len]
        attn_weights = tf.nn.softmax(scores, axis=1)  # [batch_size, enc_len]，注意力权重使用softmax归一化
        
        # 计算上下文向量
        context = tf.matmul(tf.expand_dims(attn_weights, 1), enc_out)  # [batch_size, 1, hidden]
        context = tf.squeeze(context, axis=1)  # [batch_size, hidden]
        
        # 合并解码器输出和上下文向量，预测下一个token
        combined = tf.concat([output, context], axis=-1)  # [batch_size, 2*hidden]
        attn_output = self.dense_attn(combined)  # [batch_size, hidden]
        logits = self.dense(attn_output)  # [batch_size, v_sz]
        
        # 返回预测的token和新状态
        return tf.argmax(logits, axis=-1, output_type=tf.int32), new_state

# # Loss函数以及训练逻辑

@tf.function
def compute_loss(logits, labels):
    """计算交叉熵损失"""
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
    losses = tf.reduce_mean(losses)
    return losses

@tf.function
def train_one_step(model, optimizer, enc_x, dec_x, y):
    """执行一步训练，计算损失并更新参数"""
    with tf.GradientTape() as tape:
        logits = model(enc_x, dec_x)  # 前向传播
        loss = compute_loss(logits, y)  # 计算损失

    # 计算梯度并应用更新
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def train(model, optimizer, seqlen):
    """训练模型指定步数"""
    loss = 0.0
    for step in range(2000):
        # 生成一批训练数据
        batched_examples, enc_x, dec_x, y = get_batch(32, seqlen)
        loss = train_one_step(model, optimizer, enc_x, dec_x, y)
        
        # 每500步打印一次损失
        if step % 500 == 0:
            print('step', step, ': loss', loss.numpy())
    return loss

# # 训练迭代

optimizer = optimizers.Adam(0.0005)  # 优化器
model = mySeq2SeqModel()  # 实例化模型
train(model, optimizer, seqlen=20)  # 训练模型，序列长度为20

# 测试模型逆置能力
# 首先要先对输入的一个字符串进行encode，然后在用decoder解码出逆置的字符串
# 
# 测试阶段跟训练阶段的区别在于，在训练的时候decoder的输入是给定的，而在预测的时候我们需要一步步生成下一步的decoder的输入

def sequence_reversal():
    """测试模型的序列逆置能力"""

    def decode(init_state, steps, enc_out):
        """自回归解码生成序列"""
        b_sz = tf.shape(init_state[0])[0]
        cur_token = tf.zeros(shape=[b_sz], dtype=tf.int32)  # 初始token为0（起始标记）
        state = init_state
        collect = []
        
        # 逐token生成序列
        for i in range(steps):
            cur_token, state = model.get_next_token(cur_token, state, enc_out)
            collect.append(tf.expand_dims(cur_token, axis=-1))
        
        # 将索引转换回字符串
        out = tf.concat(collect, axis=-1).numpy()
        out = [''.join([chr(idx+ord('A')-1) for idx in exp]) for exp in out]
        return out


    # 生成测试数据
    batched_examples, enc_x, _, _ = get_batch(32, 20)
    # 编码输入序列
    enc_out, state = model.encode(enc_x)
    # 解码生成逆置序列
    return decode(state, enc_x.get_shape()[-1], enc_out), batched_examples


def is_reverse(seq, rev_seq):
    """检查rev_seq是否是seq的逆置"""
    rev_seq_rev = ''.join([i for i in reversed(list(rev_seq))])
    if seq == rev_seq_rev:
        return True
    else:
        return False

# 测试模型性能
print([is_reverse(*item) for item in list(zip(*sequence_reversal()))])
print(list(zip(*sequence_reversal())))
