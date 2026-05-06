# !/usr/bin/env python
# coding: utf-8

# # 序列逆置
# 使用sequence to sequence 模型将一个字符串序列逆置。
# 例如 `OIMESIQFIQ` 逆置成 `QIFQISEMIO`(下图来自网络，是一个sequence to sequence 模型示意图 )
# ![seq2seq](./seq2seq.png)

# In[1]:

# 标准库（Python内置模块，按字母顺序排列）
import collections  # 导入Python标准库中的collections模块
import os # 导入os库，用于与操作系统交互
import sys # 导入sys库，用于系统相关参数和函数
import json
from pathlib import Path

# 第三方库（按字母顺序排列，优先导入独立库，再导入子模块）
import numpy as np# 导入NumPy库（科学计算基础库）
                    # 提供多维数组操作、数学函数、线性代数等功能
                    # 常用于数据预处理、模型输入构建和结果分析
import tensorflow as tf # 导入TensorFlow深度学习框架
from tensorflow import keras # 从TensorFlow导入Keras高级API
from tensorflow.keras import datasets, layers, optimizers # 导入Keras的子模块
# 同一库的子模块合并导入，按字母顺序排列
import random # 导入随机数生成模块，用于生成随机数、随机序列等
import string # 导入字符串常量模块，提供常用的字符串集合（如字母表、数字等）

# 运行参数（支持环境变量覆盖）
SEQREV_SEED = int(os.getenv("SEQREV_SEED", "42"))
SEQREV_TRAIN_STEPS = int(os.getenv("SEQREV_TRAIN_STEPS", "3000"))
SEQREV_TRAIN_BATCH = int(os.getenv("SEQREV_TRAIN_BATCH", "32"))
SEQREV_SEQ_LEN = int(os.getenv("SEQREV_SEQ_LEN", "20"))
SEQREV_LOG_INTERVAL = int(os.getenv("SEQREV_LOG_INTERVAL", "500"))
SEQREV_TEST_BATCH = int(os.getenv("SEQREV_TEST_BATCH", "32"))
SEQREV_TEST_LEN = int(os.getenv("SEQREV_TEST_LEN", "10"))
SEQREV_REPORT_OUT = os.getenv("SEQREV_REPORT_OUT", "sequence_reversal_report.json")
SEQREV_SHOW_SAMPLES = int(os.getenv("SEQREV_SHOW_SAMPLES", "10"))
SEQREV_DEBUG_BATCH = os.getenv("SEQREV_DEBUG_BATCH", "0") == "1"

# 固定随机种子，保证可复现
random.seed(SEQREV_SEED)
np.random.seed(SEQREV_SEED)
tf.random.set_seed(SEQREV_SEED)

# ## 玩具序列数据生成
# 生成只包含[A-Z]的字符串，并且将encoder输入以及decoder输入以及decoder输出准备好（转成index）

# In[2]:


def random_string(length):
    """
    生成一个随机字符串，字符范围为大写字母 A-Z
    参数:
        length (int): 要生成的字符串长度。
    返回:
        str: 随机生成的字符串。
    """
    # 步骤 1：定义可用字符集（大写英文字母）
    letters = string.ascii_uppercase
    
    # 步骤 2：使用 random.choices() 一次性生成指定长度的随机字符串
    # random.choices() 可以直接生成包含多个随机字符的列表
    # 然后使用 ''.join() 将列表中的字符拼接成字符串
    return ''.join(random.choices(letters, k = length))

def get_batch(batch_size, length):
    # 生成batch_size个随机字符串
    batched_examples = [random_string(length) for i in range(batch_size)]
  
    # 转成索引：字母 A-Z 映射到 1-26
    enc_x = [[ord(ch) - ord('A') + 1 for ch in list(exp)] for exp in batched_examples]
  
    # 逆序
    y = [[o for o in reversed(e_idx)] for e_idx in enc_x]
  
    #等价于y = [list(reversed(e_idx)) for e_idx in enc_x]
    # 添加起始符
    dec_x = [[0] + e_idx[:-1] for e_idx in y]
  # 返回一个批次的训练数据，包含四个张量：
# 1. batched_examples: 批量处理后的原始样本（格式取决于具体实现）
# 2. enc_x: 编码器输入序列，形状为 [batch_size, enc_seq_len]
# 3. dec_x: 解码器输入序列（通常包含起始标记），形状为 [batch_size, dec_seq_len]
# 4. y: 目标输出序列（通常包含结束标记），形状为 [batch_size, dec_seq_len]
    return (batched_examples,                        # 返回一个批次的原始样本数据
            tf.constant(enc_x, dtype = tf.int32),    # 将 enc_x 转换为 TensorFlow 的 int32 类型常量张量，作为编码器（Encoder）的输入
            tf.constant(dec_x, dtype = tf.int32),    # 将 dec_x 转换为 int32 类型的张量，作为解码器（Decoder）的输入
            tf.constant(y, dtype = tf.int32))        # 将 y 转换为 int32 类型的张量，作为标签（Label）或目标输出，用于计算损失
# 测试样例输出（默认关闭，避免影响训练日志）
if SEQREV_DEBUG_BATCH:
    print(get_batch(2, 10))

###

# # 建立sequence to sequence 模型##

# In[3]:


class mySeq2SeqModel(keras.Model):
    def __init__(self):
        # 初始化父类 keras.Model，必须调用
        """初始化Seq2Seq模型组件"""
        super().__init__()

        # 词表大小为27：A-Z共26个大写字母，加上1个特殊的起始符（用0表示）
        self.v_sz = 27 # 词表大小：26个字母+1个起始符（0）


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
            # 返回每个时间步的输出
            return_sequences = True,
            # 还返回最终隐藏状态
            return_state = True
        )

        # 解码器RNN层：与编码器类似
        self.decoder = tf.keras.layers.RNN(
            self.decoder_cell,             # 指定解码器使用的RNN单元
                                           # 例如LSTMCell、GRUCell或自定义单元
          
            return_sequences = True,       # 返回完整的输出序列
                                           # 适用于序列到序列模型，每个时间步都需要输出
                                           # 输出形状: [batch_size, seq_len, units]
          
            return_state = True            # 返回最终的隐藏状态
                                           # 对于LSTM单元，返回[h_state, c_state]
                                           # 对于GRU单元，返回[h_state]
                                           # 用于传递状态到下一个解码步骤
        )

        # 全连接层：将解码器的每个时间步的输出转换为词表大小的 logits（即每个字符的预测概率分布）
        self.dense = tf.keras.layers.Dense(self.v_sz)

        
    @tf.function
    def call(self, enc_ids, dec_ids):
        '''
        序列到序列模型的完整前向传播流程：
        编码器处理输入序列 → 传递状态给解码器 → 解码器生成输出序列 → 全连接层预测

        Args:
            enc_ids: 编码器输入序列（字符索引），shape=(batch_size, enc_seq_len)
            dec_ids: 解码器输入序列（字符索引，含起始标记），shape=(batch_size, dec_seq_len)

        Returns:
            logits: 解码器每个位置的预测概率分布，shape=(batch_size, dec_seq_len, vocab_size)
        '''
        # 编码过程
        enc_emb = self.embed_layer(enc_ids)            # (batch_size, enc_seq_len, emb_dim)
        enc_out, enc_state = self.encoder(enc_emb)     # enc_out: (batch_size, enc_seq_len, enc_units)
        
        # 解码过程，使用编码器的最终状态作为初始状态
        dec_emb = self.embed_layer(dec_ids)                                      # (batch_size, dec_seq_len, emb_dim)
        dec_out, dec_state = self.decoder(dec_emb, initial_state=enc_state)      # dec_out: (batch_size, dec_seq_len, dec_units)
        
        # 计算logits 
        logits = self.dense(dec_out)  # (batch_size, dec_seq_len, vocab_size)
      
        # 返回模型预测的logits值，通常后续会通过softmax计算概率
        # 可通过argmax获取预测的词索引：pred_ids = tf.argmax(logits, axis=-1)
        return logits
    
    
    @tf.function
    def encode(self, enc_ids):
        # shape(b_sz, len, emb_sz)，通过嵌入层将token ID转换为词向量，输出形状: (batch_size, sequence_length, embedding_size)
        enc_emb = self.embed_layer(enc_ids) 
        # 使用编码器处理嵌入向量，获取编码器输出和最终状态
        enc_out, enc_state = self.encoder(enc_emb)

        # 返回编码器最后一个时间步的输出和最终状态
        # 返回完整的编码器输出和最终状态
        return enc_out, enc_state


    def get_next_token(self, x, state):
        # 将输入token通过嵌入层转换为密集向量表示
        x_embed = self.embed_layer(x)  # (B, E)

        # 通过RNN单元计算输出和更新状态（移除了未实现的注意力机制）
        output, new_state = self.decoder_cell(x_embed, [state])

        # 通过全连接层计算logits
        logits = self.dense(output)  # (B, V)

        # 获取预测的下一个token
        next_token = tf.argmax(logits, axis=-1, output_type=tf.int32)  # (B,)

        return next_token, new_state[0]  # 返回单个状态向量


# # Loss函数以及训练逻辑

# In[4]:

# 定义了一个使用TensorFlow的@tf.function装饰器的函数compute_loss，用于计算模型预测的损失值
@tf.function
def compute_loss(logits, labels):
    """计算交叉熵损失
    数学公式:
        loss = -1/N * Σ_{i=1}^N log(exp(logits[i,label[i]]) / Σ_j exp(logits[i,j]))
    """
    # 计算稀疏交叉熵损失
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
    # 计算平均损失
    losses = tf.reduce_mean(losses)
    return losses
  
# 定义了一个使用TensorFlow的@tf.function装饰器的函数train_one_step，用于执行一个训练步骤
@tf.function  # 将函数编译为TensorFlow计算图，提升性能
def train_one_step(model, optimizer, enc_x, dec_x, y):
    """执行一次训练步骤（前向传播+反向传播）"""
    # 自动记录梯度
    with tf.GradientTape() as tape:
        # 前向传播获取预测值
        logits = model(enc_x, dec_x)
        # 计算预测值与标签的损失
        loss = compute_loss(logits, y)

    # 计算梯度（自动微分）
    grads = tape.gradient(loss, model.trainable_variables)
    
    # 应用梯度更新模型参数
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # 返回当前步骤的损失值
    return loss
def train(model, optimizer, seqlen, steps=3000, batch_size=32, log_interval=500):
    """训练过程"""
    # 初始化训练指标
    loss = 0.0 # 记录loss值 (初始为0)
    for step in range(steps):
        # 获取训练batch数据:
        # - batched_examples: 原始样本 (用于调试/可视化)
        # - enc_x: 编码器输入序列 [batch_size, seqlen]
        # - dec_x: 解码器输入序列 [batch_size, seqlen] 
        # - y: 目标输出序列 [batch_size, seqlen]
        batched_examples, enc_x, dec_x, y = get_batch(batch_size, seqlen)
        
        # 执行单步训练并返回当前loss
        loss = train_one_step(model, optimizer, enc_x, dec_x, y)
        
        # 每500步计算并打印训练进度和准确率
        if step == 0 or (step + 1) % log_interval == 0:
            # 计算训练准确率
            # 使用模型对当前批次的输入数据进行预测，得到logits
            logits = model(enc_x, dec_x)
          
            # 获取预测结果，通过argmax获取概率最高的类别索引
            preds = tf.argmax(logits, axis=-1, output_type=tf.int32)
          
            # 计算准确率，比较预测结果与真实标签是否一致，并计算平均值
            acc = tf.reduce_mean(tf.cast(tf.equal(preds, y), tf.float32))

            # 打印当前步数、损失和准确率
            print(f'step {step + 1}: loss={loss.numpy():.4f}, acc={acc.numpy():.4f}')
    return loss
# loss.numpy(): 将TensorFlow/PyTorch张量转换为NumPy数组并获取标量值

# # 训练迭代

# In[5]:
def sequence_reversal(model, batch_size=32, length=10):
    """测试阶段：对一个字符串执行encode，然后逐步decode得到逆序结果
    流程说明:
    1. 内部定义decode函数：用于执行自回归解码过程
        - 从起始标记开始，逐步生成每个字符
        - 使用模型预测下一个token
        - 收集所有生成的token形成最终输出
    2. 获取测试数据: 生成一批随机字符串样本
    3. 编码输入序列: 提取输入序列的特征表示和初始状态
    4. 解码生成逆序: 从初始状态开始逐步生成逆序序列
    5. 返回: 包含逆序结果和原始序列的元组
    
    返回格式:
        (decoded_strings, original_strings)
        decoded_strings: 模型生成的逆序字符串列表
        original_strings: 原始输入字符串列表
    """
    def decode(init_state, steps=10):
        # 获取批次大小
        batch_size = tf.shape(init_state)[0]
        # 起始 token（全为 0）
        cur_token = tf.zeros(shape=[batch_size], dtype=tf.int32)
        # 初始化状态为编码器输出的状态
        state = init_state
        # 存储每一步生成的token
        collect = []
        # 逐步解码生成序列
        for i in range(steps):
            # 获取下一个token预测和更新后的状态
            cur_token, state = model.get_next_token(cur_token, state)
            # 收集每一步生成的 token
            collect.append(tf.expand_dims(cur_token, axis=-1))
        # 拼接输出序列
        out = tf.concat(collect, axis = -1).numpy()
        # 将一个数值列表转换为对应的字母字符串
        out = [''.join([chr(idx+ord('A')-1) for idx in exp]) for exp in out] 
        return out# 返回解码后的字符串列表
    

    # 生成一批测试数据（32个样本，每个序列长度10）
    batched_examples, enc_x, _, _ = get_batch(batch_size, length)
    # 对输入序列进行编码
    _, state = model.encode(enc_x)
    # 解码生成逆序序列，步数等于输入序列长度
    return decode(state, enc_x.get_shape()[-1]), batched_examples

def is_reverse(seq, rev_seq):
    """检查 rev_seq 是否为 seq 的逆序"""
    # 反转rev_seq并与原始seq比较
    #rev_seq_rev = ''.join([i for i in reversed(list(rev_seq))])
    #if seq == rev_seq_rev:
    # 使用字符串切片来反转 rev_seq，并与 seq 比较
    if seq == rev_seq[::-1]:
        return True # 返回 True 表示预测结果与真实逆序相符
    else:
        return False
def evaluate_sequence_reversal(model, batch_size=32, length=10, show_samples=10):
    """评估模型逆序能力并返回准确率与样例。"""
    pred_list, src_list = sequence_reversal(model, batch_size=batch_size, length=length)
    pairs = list(zip(src_list, pred_list))
    flags = [is_reverse(src, pred) for src, pred in pairs]
    acc = float(np.mean(flags))
    print(f"\nreverse_accuracy: {acc:.4f}")
    for src, pred in pairs[:show_samples]:
        print(f"src={src} pred={pred} ok={is_reverse(src, pred)}")
    return acc, pairs, flags


if __name__ == "__main__":
    if (SEQREV_TRAIN_STEPS <= 0 or SEQREV_TRAIN_BATCH <= 0 or SEQREV_SEQ_LEN <= 0
            or SEQREV_LOG_INTERVAL <= 0 or SEQREV_TEST_BATCH <= 0 or SEQREV_TEST_LEN <= 0):
        raise ValueError("SEQREV 参数必须为正整数")

    optimizer = optimizers.Adam(0.0005)
    model = mySeq2SeqModel()

    print("开始训练...")
    final_loss = train(
        model,
        optimizer,
        seqlen=SEQREV_SEQ_LEN,
        steps=SEQREV_TRAIN_STEPS,
        batch_size=SEQREV_TRAIN_BATCH,
        log_interval=SEQREV_LOG_INTERVAL,
    )

    print("\n开始评估...")
    reverse_acc, pairs, flags = evaluate_sequence_reversal(
        model,
        batch_size=SEQREV_TEST_BATCH,
        length=SEQREV_TEST_LEN,
        show_samples=SEQREV_SHOW_SAMPLES,
    )

    report_path = Path(SEQREV_REPORT_OUT)
    if not report_path.is_absolute():
        report_path = Path(__file__).resolve().parent / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "seed": SEQREV_SEED,
        "train_steps": SEQREV_TRAIN_STEPS,
        "train_batch": SEQREV_TRAIN_BATCH,
        "train_seq_len": SEQREV_SEQ_LEN,
        "log_interval": SEQREV_LOG_INTERVAL,
        "test_batch": SEQREV_TEST_BATCH,
        "test_len": SEQREV_TEST_LEN,
        "final_loss": float(final_loss.numpy()),
        "reverse_accuracy": reverse_acc,
        "sample_pairs": [{"src": s, "pred": p, "ok": bool(is_reverse(s, p))} for s, p in pairs[:SEQREV_SHOW_SAMPLES]],
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("评估报告已保存:", report_path.resolve())
