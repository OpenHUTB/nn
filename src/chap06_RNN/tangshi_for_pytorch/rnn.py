import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

#对神经网络中的线性层（Linear）进行权重初始化
def weights_init(m):
    classname = m.__class__.__name__  #   obtain the class name
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out)) # 计算权重初始化范围
        m.weight.data.uniform_(-w_bound, w_bound) # 均匀分布初始化权重
        m.bias.data.fill_(0) # 偏置置零
        print("inital  linear weight ")


# 定义词嵌入模块
class word_embedding(nn.Module):
    def __init__(self, vocab_length, embedding_dim):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_length, embedding_dim)
        # PyTorch默认使用均匀分布初始化，范围为[-sqrt(1/dim), sqrt(1/dim)]
        # 如果需要特定范围，可以使用：
        # nn.init.uniform_(self.word_embedding.weight, -1.0, 1.0)

    def forward(self, input_sentence):
        """
        :param input_sentence: 一个包含词索引的张量
        :return: 一个包含对应词向量的张量
        """
        return self.word_embedding(input_sentence)


# 定义RNN模型
class RNN_model(nn.Module):
    def __init__(self, batch_sz, vocab_len, word_embedding, embedding_dim, lstm_hidden_dim):
        super(RNN_model, self).__init__()

        # 模型组件初始化
        self.word_embedding_lookup = word_embedding  # 使用外部定义的词嵌入模块
        self.batch_size = batch_sz  # 批处理大小
        self.vocab_length = vocab_len  #词汇表大小
        self.word_embedding_dim = embedding_dim  #词向量维度
        self.lstm_dim = lstm_hidden_dim  # LSTM隐藏状态维度

        # 定义LSTM层
        # input_size=embedding_dim：输入的特征维度，即词嵌入维度
        # hidden_size=lstm_hidden_dim：LSTM隐藏状态的维度
        # num_layers=2：堆叠两层LSTM
        # batch_first=False：输入张量的维度顺序为(seq_len, batch, input_size)
        self.rnn_lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_dim, num_layers=2, batch_first=False)

        # 定义全连接层，将LSTM的输出映射到词汇表大小
        self.fc = nn.Linear(lstm_hidden_dim, vocab_len)

        # 调用权重初始化函数（需要你自定义weights_init函数）
        self.apply(weights_init)

        # 定义log softmax激活函数作为输出层
        self.softmax = nn.LogSoftmax(dim=1)  # 明确指定维度，避免警告

    def forward(self, sentence, is_test=False):
        # 查找词向量并调整形状为 (1, 序列长度, 嵌入维度)
        batch_input = self.word_embedding_lookup(sentence).view(1, -1, self.word_embedding_dim)

        # 初始化LSTM的初始隐藏状态和记忆细胞为0
        h0 = torch.zeros(2, batch_input.size(1), self.lstm_dim)  # (层数, 批大小, 隐藏维度)
        c0 = torch.zeros(2, batch_input.size(1), self.lstm_dim)

        # 如果有GPU可用，将张量移动到GPU
        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()
            batch_input = batch_input.cuda()

        # 前向传播：将输入送入LSTM
        output, (hn, cn) = self.rnn_lstm(batch_input, (h0, c0))

        # 将LSTM输出转换为二维张量以输入全连接层
        out = output.contiguous().view(-1, self.lstm_dim)

        # 通过全连接层和ReLU激活
        out = F.relu(self.fc(out))

        # 使用log softmax获得输出分布
        out = self.softmax(out)

        # 如果是测试模式，仅返回最后一个时间步的预测结果
        if is_test:
            prediction = out[-1, :].view(1, -1)
            output = prediction
        else:
            output = out

        return output

