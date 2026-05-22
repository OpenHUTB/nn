#!/usr/bin/env python
# coding: utf-8
"""
基于 RNN 的唐诗生成模型。

使用 SimpleRNN 学习唐诗的韵律和结构，实现自动诗歌生成。
"""

import collections
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers

# 特殊标记
START_TOKEN = '<START>'
END_TOKEN = '<END>'


def process_dataset(filename: str) -> Tuple[List[Tuple[List[int], int]], Dict[str, int], Dict[int, str]]:
    """处理诗歌数据集，构建词汇表和数字索引的诗歌数据。

    Args:
        filename: 诗歌文本文件路径，格式为"标题:内容"

    Returns:
        instances: 数字索引化的诗歌列表，每个元素为 (id序列, 序列长度)
        word2id: 词语到数字id的映射字典
        id2word: 数字id到词语的映射字典
    """
    examples = []

    with open(filename, 'r', encoding='utf-8') as fd:
        for line in fd:
            parts = line.strip().split(':')
            content = ''.join(parts[1:])
            tokens = [START_TOKEN] + list(content) + [END_TOKEN]
            if len(tokens) <= 200:
                examples.append(tokens)

    # 统计词频并按频率排序
    counter = collections.Counter(token for poem in examples for token in poem)
    sorted_words = sorted(counter, key=counter.get, reverse=True)

    # 构建词汇表：PAD=0, UNK=1, 其余按词频排列
    words = ('PAD', 'UNK') + tuple(sorted_words)
    word2id = dict(zip(words, range(len(words))))
    id2word = {v: k for k, v in word2id.items()}

    # 将诗歌转换为数字id序列
    indexed_examples = [[word2id[w] for w in poem] for poem in examples]
    seqlen = [len(e) for e in indexed_examples]
    instances = list(zip(indexed_examples, seqlen))

    return instances, word2id, id2word


def poem_dataset(filename: str = '../poems.txt'):
    """创建诗歌数据集的 TensorFlow Dataset 对象。

    Args:
        filename: 诗歌文本文件路径

    Returns:
        ds: 处理好的 tf.data.Dataset 对象
        word2id: 词语到id的映射
        id2word: id到词语的映射
    """
    instances, word2id, id2word = process_dataset(filename)

    ds = tf.data.Dataset.from_generator(
        lambda: instances,
        output_types=(tf.int64, tf.int64),
        output_shapes=(tf.TensorShape([None]), tf.TensorShape([])),
    )

    ds = ds.shuffle(buffer_size=10240)
    ds = ds.padded_batch(100, padded_shapes=(tf.TensorShape([None]), tf.TensorShape([])))
    # 准备语言模型的输入输出对：输入是前n-1个词，输出是后n-1个词
    ds = ds.map(lambda x, seqlen: (x[:, :-1], x[:, 1:], seqlen - 1))

    return ds, word2id, id2word


class RNNModel(tf.keras.Model):
    """基于 SimpleRNN 的诗歌生成模型。"""

    def __init__(self, word2id: Dict[str, int]):
        super().__init__()
        vocab_size = len(word2id)

        self.embedding = layers.Embedding(
            vocab_size, 64, batch_input_shape=[None, None]
        )
        self.rnn_cell = layers.SimpleRNNCell(128)
        self.rnn_layer = layers.RNN(self.rnn_cell, return_sequences=True)
        self.dense = layers.Dense(vocab_size)

    @tf.function
    def call(self, inp_ids: tf.Tensor) -> tf.Tensor:
        """前向传播。

        Args:
            inp_ids: 输入词id序列，形状 (batch_size, seq_len)

        Returns:
            logits: 每个位置的下一个词预测，形状 (batch_size, seq_len, vocab_size)
        """
        x = self.embedding(inp_ids)
        rnn_out = self.rnn_layer(x)
        return self.dense(rnn_out)

    @tf.function
    def get_next_token(self, x: tf.Tensor, state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """生成下一个词（用于文本生成阶段）。

        Args:
            x: 当前词id，形状 (batch_size,)
            state: RNN 的隐藏状态

        Returns:
            out: 预测的下一个词id
            state: 更新后的 RNN 状态
        """
        inp_emb = self.embedding(x)
        h, state = self.rnn_cell(inp_emb, state)
        logits = self.dense(h)
        out = tf.argmax(logits, axis=-1)
        return out, state


def create_mask(lengths: tf.Tensor, max_len: tf.Tensor) -> tf.Tensor:
    """创建序列掩码，用于处理变长序列。

    Args:
        lengths: 包含序列长度的张量
        max_len: 最大序列长度

    Returns:
        布尔掩码张量
    """
    return tf.sequence_mask(tf.reshape(lengths, [-1]), maxlen=max_len)


def masked_mean(values: tf.Tensor, lengths: tf.Tensor, axis: int) -> tf.Tensor:
    """沿指定维度计算掩码后的平均值（忽略填充部分）。

    Args:
        values: 需要求平均的张量
        lengths: 每个序列的实际长度
        axis: 沿哪个维度求平均

    Returns:
        掩码后的平均值
    """
    mask = tf.sequence_mask(lengths, maxlen=tf.shape(values)[axis])

    # 调整 mask 形状以匹配广播
    rank_diff = len(values.shape) - len(mask.shape)
    if rank_diff > 0:
        mask = tf.reshape(mask, tf.concat([tf.shape(mask), [1] * rank_diff], axis=0))

    mask = tf.cast(mask, dtype=values.dtype)
    masked_values = values * mask

    sum_val = tf.reduce_sum(masked_values, axis=axis, keepdims=False)
    count = tf.cast(tf.reshape(lengths, tf.shape(sum_val)), tf.float32) + 1e-30

    return sum_val / count


@tf.function
def compute_loss(logits: tf.Tensor, labels: tf.Tensor, seqlen: tf.Tensor) -> tf.Tensor:
    """计算序列的交叉熵损失（考虑变长序列）。

    Args:
        logits: 模型输出，形状 (batch_size, seq_len, vocab_size)
        labels: 真实标签，形状 (batch_size, seq_len)
        seqlen: 每个序列的实际长度

    Returns:
        平均损失值
    """
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    losses = masked_mean(losses, seqlen, axis=1)
    return tf.reduce_mean(losses)


@tf.function
def train_one_step(model: RNNModel, optimizer, x: tf.Tensor, y: tf.Tensor, seqlen: tf.Tensor) -> tf.Tensor:
    """执行一步训练。

    Args:
        model: 诗歌生成模型
        optimizer: 优化器
        x: 输入序列
        y: 目标序列
        seqlen: 序列长度

    Returns:
        当前步骤的损失值
    """
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(logits, y, seqlen)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def train_epoch(epoch: int, model: RNNModel, optimizer, dataset) -> float:
    """训练一个 epoch。

    Args:
        epoch: 当前 epoch 编号
        model: 诗歌生成模型
        optimizer: 优化器
        dataset: 训练数据集

    Returns:
        最后一个批次的损失值
    """
    loss = 0.0
    for step, (x, y, seqlen) in enumerate(dataset):
        loss = train_one_step(model, optimizer, x, y, seqlen)
        if step % 500 == 0:
            print(f'epoch {epoch}: loss {loss.numpy():.4f}')
    return loss


def generate_poem(model: RNNModel, word2id: Dict[str, int], id2word: Dict[int, str], max_len: int = 50) -> str:
    """使用训练好的 RNN 模型生成诗歌。

    Args:
        model: 训练好的诗歌生成模型
        word2id: 词语到id的映射字典
        id2word: id到词语的映射字典
        max_len: 生成诗歌的最大长度

    Returns:
        生成的诗歌字符串（不包含开始和结束标记）
    """
    state = tf.random.normal(shape=(1, 128), stddev=0.5)
    cur_token = tf.constant([word2id[START_TOKEN]], dtype=tf.int32)
    generated_tokens = []

    for _ in range(max_len):
        cur_token, state = model.get_next_token(cur_token, state)
        token_id = cur_token.numpy()[0]
        generated_tokens.append(token_id)

        if id2word[token_id] == END_TOKEN:
            break

    # 去除开始和结束标记
    return ''.join(id2word[t] for t in generated_tokens[1:-1])


if __name__ == '__main__':
    # 加载数据集
    train_ds, word2id, id2word = poem_dataset()

    # 初始化模型和优化器
    model = RNNModel(word2id)
    optimizer = optimizers.Adam(0.0005)

    # 训练模型
    for epoch in range(10):
        loss = train_epoch(epoch, model, optimizer, train_ds)

    # 生成诗歌示例
    print('\n生成的诗歌：')
    print(generate_poem(model, word2id, id2word))
    print(generate_poem(model, word2id, id2word))
