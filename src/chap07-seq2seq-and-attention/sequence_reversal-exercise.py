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

    letters = string.ascii_uppercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def get_batch(batch_size, length):
    batched_examples = [randomString(length) for i in range(batch_size)]
    enc_x = [[ord(ch)-ord('A')+1 for ch in list(exp)] for exp in batched_examples]
    y = [[o for o in reversed(e_idx)] for e_idx in enc_x]
    dec_x = [[0]+e_idx[:-1] for e_idx in y]
    return (batched_examples, tf.constant(enc_x, dtype=tf.int32), 
            tf.constant(dec_x, dtype=tf.int32), tf.constant(y, dtype=tf.int32))
print(get_batch(2, 10))


# # 建立sequence to sequence 模型

# In[3]:


class mySeq2SeqModel(keras.Model):
    def __init__(self):
        super(mySeq2SeqModel, self).__init__()
        self.v_sz=27
        self.embed_layer = tf.keras.layers.Embedding(self.v_sz, 64, 
                                                    batch_input_shape=[None, None])
        
        self.encoder_cell = tf.keras.layers.SimpleRNNCell(128)
        self.decoder_cell = tf.keras.layers.SimpleRNNCell(128)
        
        self.encoder = tf.keras.layers.RNN(self.encoder_cell, 
                                           return_sequences=True, return_state=True)
        self.decoder = tf.keras.layers.RNN(self.decoder_cell, 
                                           return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(self.v_sz)
        
    @tf.function
    def call(self, enc_ids, dec_ids):
        '''
        完成sequence2sequence 模型的搭建，模块已经在`__init__`函数中定义好
        '''
        return logits
    
    
#     @tf.function
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


@tf.function
def compute_loss(logits, labels):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
    losses = tf.reduce_mean(losses)
    return losses

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
# 
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




