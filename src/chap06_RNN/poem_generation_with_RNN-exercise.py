#!/usr/bin/env python
# coding: utf-8

# # 诗歌生成

# # 数据处理

# In[1]:


import numpy as np
import tensorflow as tf
import collections
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers, optimizers, datasets

start_token = 'bos'
end_token = 'eos'

def process_dataset(fileName):
    examples = []
    with open(fileName, 'r') as fd:
        for line in fd:
            outs = line.strip().split(':')
            content = ''.join(outs[1:])
            ins = [start_token] + list(content) + [end_token] 
            if len(ins) > 200:
                continue
            examples.append(ins)
            
    counter = collections.Counter()
    for e in examples:
        for w in e:
            counter[w] += 1
    
    sorted_counter = sorted(counter.items(), key=lambda x: -x[1])  # 排序
    words, _ = zip(*sorted_counter)
    words = ('PAD', 'UNK') + words[:len(words)]
    word2id = dict(zip(words, range(len(words))))
    id2word = {word2id[k]:k for k in word2id}
    
    indexed_examples = [[word2id[w] for w in poem]
                        for poem in examples]
    seqlen = [len(e) for e in indexed_examples]
    
    instances = list(zip(indexed_examples, seqlen))
    
    return instances, word2id, id2word

def poem_dataset():
    instances, word2id, id2word = process_dataset('../poems.txt')
    ds = tf.data.Dataset.from_generator(lambda: [ins for ins in instances], 
                                            (tf.int64, tf.int64), 
                                            (tf.TensorShape([None]),tf.TensorShape([])))
    ds = ds.shuffle(buffer_size=10240)
    ds = ds.padded_batch(100, padded_shapes=(tf.TensorShape([None]),tf.TensorShape([])))
    ds = ds.map(lambda x, seqlen: (x[:, :-1], x[:, 1:], seqlen-1))
    return ds, word2id, id2word


# # 模型代码， 完成建模代码

# In[2]:


class myRNNModel(keras.Model):
    def __init__(self, w2id):
        super(myRNNModel, self).__init__()
        self.v_sz = len(w2id)
        self.embed_layer = tf.keras.layers.Embedding(self.v_sz, 64, 
                                                    batch_input_shape=[None, None])
        
        self.rnncell = tf.keras.layers.SimpleRNNCell(128)
        self.rnn_layer = tf.keras.layers.RNN(self.rnncell, return_sequences=True)
        self.dense = tf.keras.layers.Dense(self.v_sz)
        
    @tf.function
    def call(self, inp_ids):
        '''
        此处完成建模过程，可以参考Learn2Carry
        '''
        return logits
    
    @tf.function
    def get_next_token(self, x, state):
        '''
        shape(x) = [b_sz,] 
        '''
    
        inp_emb = self.embed_layer(x) #shape(b_sz, emb_sz)
        h, state = self.rnncell.call(inp_emb, state) # shape(b_sz, h_sz)
        logits = self.dense(h) # shape(b_sz, v_sz)
        out = tf.argmax(logits, axis=-1)
        return out, state


# ## 一个计算sequence loss的辅助函数，只需了解用途。

# In[3]:


def mkMask(input_tensor, maxLen):
    shape_of_input = tf.shape(input_tensor)
    shape_of_output = tf.concat(axis=0, values=[shape_of_input, [maxLen]])

    oneDtensor = tf.reshape(input_tensor, shape=(-1,))
    flat_mask = tf.sequence_mask(oneDtensor, maxlen=maxLen)
    return tf.reshape(flat_mask, shape_of_output)


def reduce_avg(reduce_target, lengths, dim):
    """
    Args:
        reduce_target : shape(d_0, d_1,..,d_dim, .., d_k)
        lengths : shape(d0, .., d_(dim-1))
        dim : which dimension to average, should be a python number
    """
    shape_of_lengths = lengths.get_shape()
    shape_of_target = reduce_target.get_shape()
    if len(shape_of_lengths) != dim:
        raise ValueError(('Second input tensor should be rank %d, ' +
                         'while it got rank %d') % (dim, len(shape_of_lengths)))
    if len(shape_of_target) < dim+1 :
        raise ValueError(('First input tensor should be at least rank %d, ' +
                         'while it got rank %d') % (dim+1, len(shape_of_target)))

    rank_diff = len(shape_of_target) - len(shape_of_lengths) - 1
    mxlen = tf.shape(reduce_target)[dim]
    mask = mkMask(lengths, mxlen)
    if rank_diff!=0:
        len_shape = tf.concat(axis=0, values=[tf.shape(lengths), [1]*rank_diff])
        mask_shape = tf.concat(axis=0, values=[tf.shape(mask), [1]*rank_diff])
    else:
        len_shape = tf.shape(lengths)
        mask_shape = tf.shape(mask)
    lengths_reshape = tf.reshape(lengths, shape=len_shape)
    mask = tf.reshape(mask, shape=mask_shape)

    mask_target = reduce_target * tf.cast(mask, dtype=reduce_target.dtype)

    red_sum = tf.reduce_sum(mask_target, axis=[dim], keepdims=False)
    red_avg = red_sum / (tf.cast(lengths_reshape, dtype=tf.float32) + 1e-30)
    return red_avg


# # 定义loss函数，定义训练函数

# In[4]:


@tf.function
def compute_loss(logits, labels, seqlen):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
    losses = reduce_avg(losses, seqlen, dim=1)
    return tf.reduce_mean(losses)

@tf.function
def train_one_step(model, optimizer, x, y, seqlen):
    '''
    完成一步优化过程，可以参考之前做过的模型
    '''
    return loss

def train(epoch, model, optimizer, ds):
    loss = 0.0
    accuracy = 0.0
    for step, (x, y, seqlen) in enumerate(ds):
        loss = train_one_step(model, optimizer, x, y, seqlen)

        if step % 500 == 0:
            print('epoch', epoch, ': loss', loss.numpy())

    return loss


# # 训练优化过程

# In[5]:


optimizer = optimizers.Adam(0.0005)
train_ds, word2id, id2word = poem_dataset()
model = myRNNModel(word2id)

for epoch in range(10):
    loss = train(epoch, model, optimizer, train_ds)


# # 生成过程

# In[74]:


def gen_sentence():
    state = [tf.random.normal(shape=(1, 128), stddev=0.5), tf.random.normal(shape=(1, 128), stddev=0.5)]
    cur_token = tf.constant([word2id['bos']], dtype=tf.int32)
    collect = []
    for _ in range(50):
        cur_token, state = model.get_next_token(cur_token, state)
        collect.append(cur_token.numpy()[0])
    return [id2word[t] for t in collect]
print(''.join(gen_sentence()))


# In[ ]:





# In[ ]:




