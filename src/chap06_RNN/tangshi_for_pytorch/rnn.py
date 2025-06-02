import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

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


class word_embedding(nn.Module):
    def __init__(self,vocab_length , embedding_dim):
        super(word_embedding, self).__init__()
        w_embeding_random_intial = np.random.uniform(-1,1,size=(vocab_length ,embedding_dim))
        self.word_embedding = nn.Embedding(vocab_length,embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(w_embeding_random_intial))
    def forward(self,input_sentence):
        """
        :param input_sentence:  a tensor ,contain several word index.
        :return: a tensor ,contain word embedding tensor
        """
        sen_embed = self.word_embedding(input_sentence)
        return sen_embed

class RNN_model(nn.Module):
    def __init__(self, batch_sz ,vocab_len ,word_embedding,embedding_dim, lstm_hidden_dim):
        super(RNN_model,self).__init__()

        self.word_embedding_lookup = word_embedding
        self.batch_size = batch_sz
        self.vocab_length = vocab_len
        self.word_embedding_dim = embedding_dim
        self.lstm_dim = lstm_hidden_dim
        #input_size=embedding_dim：输入特征的维度，与嵌入向量的维度相同。
        #hidden_size=lstm_hidden_dim：LSTM隐藏状态的维度。
        #num_layers=2：LSTM层的数量，这里是2层堆叠的LSTM。
        #batch_first=False：输入和输出的张量形状为(seq_len, batch, input_size)，而不是(batch, seq_len, input_size)。这意味着序列长度是第一个维度。
        self.rnn_lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_dim, num_layers=2, batch_first=False)
        
        self.fc = nn.Linear(lstm_hidden_dim, vocab_len )
        self.apply(weights_init) # call the weights initial function.

        self.softmax = nn.LogSoftmax() # the activation function.
        # self.tanh = nn.Tanh()
    def forward(self,sentence,is_test = False):
        batch_input = self.word_embedding_lookup(sentence).view(1,-1,self.word_embedding_dim)
        # print(batch_input.size()) # print the size of the input
        # here you need to put the "batch_input"  input the self.lstm which is defined before.
        # the hidden output should be named as output, the initial hidden state and cell state set to zero.
        # ???
        h0 = torch.zeros(2, batch_input.size(1), self.lstm_dim)  # (num_layers, batch_size, hidden_dim)
        c0 = torch.zeros(2, batch_input.size(1), self.lstm_dim)  # (num_layers, batch_size, hidden_dim)
        if torch.cuda.is_available():  # If using GPU, move tensors to GPU
            h0 = h0.cuda()
            c0 = c0.cuda()
            batch_input = batch_input.cuda()
        
        output, (hn, cn) = self.rnn_lstm(batch_input, (h0, c0))  # LSTM forward pass

        out = output.contiguous().view(-1,self.lstm_dim)

        out =  F.relu(self.fc(out))

        out = self.softmax(out)

        if is_test:
            prediction = out[ -1, : ].view(1,-1)
            output = prediction
        else:
           output = out
        # print(out)
        return output

