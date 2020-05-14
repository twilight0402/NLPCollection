import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class LSTM_model(nn.Module):
    """
    多层 lstm + linear
    """
    def __init__(
        self,
        vocab_size,         # 词表的长度（总单词数）， 8188
        output_dim,         # 输出维度，分类类别数，5分类
        n_layers=2,         # lstm 层数
        pad_idx=None,       # 还不知道是啥 ？？？
        hidden_dim=128,     # lstm 的hidden数量
        embed_dim=300,      # 300维词向量
        dropout=0.1,
        bidirectional=False,
    ):
        super().__init__()  # 调用父类构造函数初始化从父类继承的变量
        num_directions = 1 if not bidirectional else 2
        # Embeddiing(总单词数， embedding维度)
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx,
        )
        # (300， 128)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(p=dropout)
        # (128 * 2 * 1)
        self.linear = nn.Linear(hidden_dim * n_layers * num_directions, output_dim)

    def forward(self, x, x_len):
        # 输入的x可以选择特定的词向量组成高维度的矩阵；输入为(batch, word_num) ，输出为 (batch, word_num, embedding_dim)
        # 实际传进来的是 (word_num, batch_size)=>(8, 64)
        x = self.embedding(x)
        # 实际得到的x (8, 64, 300)
        # Pad each sentences for a batch,
        # the final x with shape (seq_len, batch_size, embed_size)
        # 压缩， (seq_len, batch_size, embed_size)  ==>    (总单词数， embed_size)
        x = pack_padded_sequence(x, x_len)
        # h_n: (num_layers * num_directions, batch_size, hidden_size)
        # NOTE: take the last hidden state of encoder as in seq2seq architecture.
        hidden_states, (h_n, c_c) = self.lstm(x)

        hidden_states_packed = nn.utils.rnn.pad_packed_sequence(hidden_states, batch_first=False)
        # h_n_packed = nn.utils.rnn.pad_packed_sequence(h_n, batch_first=False)
        # c_c_packed = nn.utils.rnn.pad_packed_sequence(c_c, batch_first=False)

        # 交换前两个shape，这样就变成了batchfirst格式的数据了， 并返回一个保存在整块内存中的新变量
        h_n = torch.transpose(self.dropout(h_n), 0, 1).contiguous()
        # h_n:(batch_size, hidden_size * num_layers * num_directions)
        h_n = h_n.view(h_n.shape[0], -1)
        loggits = self.linear(h_n)
        return loggits
