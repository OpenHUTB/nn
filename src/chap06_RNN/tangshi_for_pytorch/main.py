import numpy as np
import collections
import torch
from torch.autograd import Variable
import torch.optim as optim

import rnn

# 定义特殊标记
start_token = 'B'  # 序列开始标记
end_token = 'E'    # 序列结束标记
batch_size = 64    # 批处理大小


def process_poems1(file_name):
    """
    处理古诗文本文件，返回诗歌的向量表示（每个字映射为索引）。

    :param file_name: 包含诗歌的文件路径，格式为 每行 "标题:内容"
    :return:
        poems_vector: 二维列表，每首诗转换为字的索引序列
        word_int_map: 字到索引的映射字典
        words: 所有字组成的元组，按频率降序排序，最后加一个空格符
    例子：[[1,2,3,4],[5,6,7,8]]
    """

    poems = []  # 存储处理后的诗歌
    with open(file_name, "r", encoding='utf-8') as f:
        for line in f.readlines():
            try:
                # 尝试按“标题:内容”格式解析
                title, content = line.strip().split(':')
                # 去除空格
                content = content.replace(' ', '')

                # 跳过包含特殊字符或起始/结束标记的诗句
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                        start_token in content or end_token in content:
                    continue
                # 跳过长度不合理的诗句
                if len(content) < 5 or len(content) > 80:
                    continue

                # 添加起始和结束标记
                content = start_token + content + end_token
                # 将处理后的诗歌内容添加到列表中
                poems.append(content)
            except ValueError:
                print("error")  # 如果行不符合“标题:内容”格式则跳过
                pass

    # 按诗的长度进行排序，便于后续按批处理
    poems = sorted(poems, key=lambda line: len(line))

    # 统计所有诗句中的字频
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]  # 拆成单字列表

    counter = collections.Counter(all_words)  # 统计每个字的出现次数
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 按频率降序排序

    # 提取所有字，按频率排列，加一个空格符用于补齐
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)

    # 构建字到索引的映射
    word_int_map = dict(zip(words, range(len(words))))

    # 将诗句转为索引序列
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]

    return poems_vector, word_int_map, words


def process_poems2(file_name):
    """
    处理诗歌文本数据，转换为向量表示
    :param file_name: 输入的文本文件名，每行为一首诗
    :return: 
        poems_vector：二维列表，第一维是诗的数量，第二维是每首诗中每个字对应的索引
        word_int_map：字到索引的映射字典
        words：包含所有字的元组，按出现频率排序

    示例：
        poems_vector = [[1, 2, 3, 4], [5, 2, 8, 7]]
    """

    poems = []  # 存储所有符合条件的诗
    with open(file_name, "r", encoding='utf-8') as f:
        for line in f.readlines():
            try:
                line = line.strip()  # 去除首尾空白符
                if line:
                    # 移除空格和常见标点符号
                    content = line.replace(' ', '').replace('，', '').replace('。', '')

                    # 过滤包含特殊字符的诗句
                    if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                            start_token in content or end_token in content:
                        continue

                    # 过滤长度不符合要求的诗句
                    if len(content) < 5 or len(content) > 80:
                        continue

                    # 添加起始符和结束符
                    content = start_token + content + end_token
                    poems.append(content)

            except ValueError:
                # 忽略读取或处理异常
                pass

    # 按诗的长度进行排序（便于后续批处理时填充对齐）
    poems = sorted(poems, key=lambda line: len(line))

    # 统计所有诗中每个字出现的频率
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]

    # 使用Counter统计词频，并按频率降序排序
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])

    # 提取所有字，添加空格字符用于填充
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)

    # 建立字到索引的映射
    word_int_map = dict(zip(words, range(len(words))))

    # 将所有诗转为索引表示
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]

    return poems_vector, word_int_map, words

def generate_batch(batch_size, poems_vec, word_to_int):
    """
    生成训练所需的批次数据（x_batches 和 y_batches）

    参数:
    - batch_size: 每个批次的样本数（即每次喂给模型多少首诗）
    - poems_vec: 所有诗歌的索引表示（列表形式，里面是每首诗的词索引序列）
    - word_to_int: 词到索引的映射字典（未在本函数中使用，但一般可用于逆映射）

    返回:
    - x_batches: 输入数据批次，每个元素是一个形状为 (batch_size, seq_len) 的输入序列列表
    - y_batches: 目标数据批次，每个元素是对应输入的下一个词序列（即标签）
    """

    n_chunk = len(poems_vec) // batch_size  # 计算可以划分成多少完整的 batch（整除部分）

    x_batches = []  # 用于存储所有输入批次
    y_batches = []  # 用于存储所有目标批次

    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size

        # 从诗歌向量集中取出一个 batch 的诗句（每首诗是一个词索引列表）
        x_data = poems_vec[start_index:end_index]

        y_data = []
        for row in x_data:
            # 构造目标序列 y：将原序列向后偏移一位，并补上最后一个词（常用于语言模型预测下一个词）
            y = row[1:]          # 将序列右移一位
            y.append(row[-1])    # 最后一个词复制一份填充，确保长度一致
            y_data.append(y)

        """
        示例：
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]  # 下一个词的预测目标
        [1,4,2,8,5]       [4,2,8,5,5]
        """

        x_batches.append(x_data)
        y_batches.append(y_data)

    return x_batches, y_batches


def run_training():
    """训练古诗生成模型"""
    # 处理数据集
    # poems_vector, word_to_int, vocabularies = process_poems2('./tangshi.txt')
    poems_vector, word_to_int, vocabularies = process_poems1('./poems.txt')
    
    # 生成batch
    print("finish  loading data")
    BATCH_SIZE = 100  # 每批次处理的样本数量

    # 设置随机种子以确保结果可复现
    torch.manual_seed(5)
    
    # 模型初始化：创建词嵌入层和RNN模型
    # 创建词嵌入层，为每个词生成100维的向量表示
    word_embedding = rnn_lstm.word_embedding(
        vocab_length=len(word_to_int) + 1, 
        embedding_dim=100
    )
    
    # 创建RNN模型，使用LSTM作为核心结构
    rnn_model = rnn_lstm.RNN_model(
        batch_sz=BATCH_SIZE,
        vocab_len=len(word_to_int) + 1,
        word_embedding=word_embedding,
        embedding_dim=100,
        lstm_hidden_dim=128
    )

    # 配置优化器和损失函数
    # optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)
    optimizer = optim.RMSprop(rnn_model.parameters(), lr=0.01)
    loss_fun = torch.nn.NLLLoss()
    
    # 如果已有训练好的模型，可以取消注释此行加载
    # rnn_model.load_state_dict(torch.load('./poem_generator_rnn'))

    # 训练循环
    for epoch in range(30):
        # 生成批次数据
        batches_inputs, batches_outputs = generate_batch(BATCH_SIZE, poems_vector, word_to_int)
        n_chunk = len(batches_inputs)
        
        # 遍历每个批次
        for batch in range(n_chunk):
            batch_x = batches_inputs[batch]
            batch_y = batches_outputs[batch]  # (batch, time_step)

            loss = 0  # 初始化批次损失
            
            # 处理批次中的每个样本
            for index in range(BATCH_SIZE):
                x = np.array(batch_x[index], dtype=np.int64)  # 样本输入序列
                y = np.array(batch_y[index], dtype=np.int64)  # 样本目标序列
                
                # 转换为PyTorch张量并调整维度
                x = Variable(torch.from_numpy(np.expand_dims(x, axis=1)))
                y = Variable(torch.from_numpy(y))
                
                # 前向传播
                pre = rnn_model(x)
                loss += loss_fun(pre, y)
                
                # 每批次的第一个样本打印预测结果用于调试
                if index == 0:
                    _, pre = torch.max(pre, dim=1)
                    print('prediction', pre.data.tolist())  # 打印预测结果
                    print('b_y       ', y.data.tolist())    # 打印真实标签
                    print('*' * 30)
            
            # 计算平均损失
            loss = loss / BATCH_SIZE
            print(f"epoch {epoch}, batch {batch}, loss: {loss.data.tolist()}")
            
            # 反向传播和参数更新
            optimizer.zero_grad()       # 梯度清零
            loss.backward()             # 反向传播计算梯度
            torch.nn.utils.clip_grad_norm(rnn_model.parameters(), 1)  # 梯度裁剪防止梯度爆炸
            optimizer.step()            # 更新模型参数

            # 每20个批次保存一次模型
            if batch % 20 == 0:
                torch.save(rnn_model.state_dict(), './poem_generator_rnn')
                print("finish  save model")


def to_word(predict, vocabs):
    """将预测的概率分布转换为对应的字
    Args:
        predict: 预测的概率分布
        vocabs: 词汇表
    Returns:
        预测概率最大的字
    """
    sample = np.argmax(predict)
    
    # 防止索引越界
    if sample >= len(vocabs):
        sample = len(vocabs) - 1
        
    return vocabs[sample]


def pretty_print_poem(poem):
    """格式化打印古诗，使其更符合阅读习惯
    Args:
        poem: 生成的古诗文本
    """
    shige = []  # 存储有效诗句的列表
    
    # 过滤掉开始和结束标记
    for w in poem:
        if w == start_token or w == end_token:
            break
        shige.append(w)
    
    # 按句号分割诗句并打印
    poem_sentences = ''.join(shige).split('。')
    for s in poem_sentences:
        if s != '' and len(s) > 1:
            print(s + '。')


def gen_poem(begin_word):
    """
    以指定字开头生成古诗
    
    Args:
        begin_word: 起始字
        
    Returns:
        poem: 生成的完整古诗
    """
    # 加载数据
    poems_vector, word_int_map, vocabularies = process_poems1('./poems.txt')
    
    # 创建模型并加载预训练权重
    word_embedding = rnn_lstm.word_embedding(
        vocab_length=len(word_int_map) + 1,
        embedding_dim=100
    )
    
    rnn_model = rnn_lstm.RNN_model(
        batch_sz=64,
        vocab_len=len(word_int_map) + 1,
        word_embedding=word_embedding,
        embedding_dim=100,
        lstm_hidden_dim=128
    )
    
    rnn_model.load_state_dict(torch.load('./poem_generator_rnn'))
    
    # 初始化生成的诗句
    poem = begin_word
    word = begin_word
    
    # 循环生成诗句，直到遇到结束标记或达到最大长度
    while word != end_token:
        # 将当前诗句转换为模型输入
        input = np.array([word_int_map[w] for w in poem], dtype=np.int64)
        input = Variable(torch.from_numpy(input))
        
        # 模型预测下一个字
        output = rnn_model(input, is_test=True)
        word = to_word(output.data.tolist()[-1], vocabularies)
        
        # 添加预测的字到诗句
        poem += word
        
        # 防止生成过长的诗句
        if len(poem) > 30:
            break
            
    return poem


# 训练模型（如果已训练好模型，请注释掉此行）
run_training()

# 测试生成功能
pretty_print_poem(gen_poem("日"))
pretty_print_poem(gen_poem("红"))
pretty_print_poem(gen_poem("山"))
pretty_print_poem(gen_poem("夜"))
pretty_print_poem(gen_poem("湖"))
pretty_print_poem(gen_poem("湖"))
pretty_print_poem(gen_poem("湖"))
pretty_print_poem(gen_poem("君"))
