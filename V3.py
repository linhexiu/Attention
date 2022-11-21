# 注意力分数
# https://zh-v2.d2l.ai/chapter_attention-mechanisms/attention-scoring-functions.html

import math
import torch
from torch import nn
from d2l import torch as d2l


def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)

        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)

        return nn.functional.softmax(X.reshape(shape), dim=-1)


# AdditiveAttention 加性注意力
class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout):
        super(AdditiveAttention, self).__init__()
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries = self.W_q(queries)  # (2, 1, 8)
        keys = self.W_k(keys)  # (2, 10, 8)

        # queries1 = queries.unsqueeze(2)
        # keys1=keys.unsqueeze(1)
        # (2, 1, 10, 8)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)

        # t=self.W_v(features)
        scores = self.W_v(features).squeeze(-1)

        self.attention_weights = masked_softmax(scores, valid_lens)

        return torch.bmm(self.dropout(self.attention_weights), values)


# data
queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))

# values.shape : (2, 10, 4)
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)

valid_lens = torch.tensor([2, 6])

attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                              dropout=0.1)

attention.eval()
attention(queries, keys, values, valid_lens)


# DotProductAttention  缩放点积注意力
class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]

        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


queries = torch.normal(0, 1, (2, 1, 2))
attention = DotProductAttention(dropout=0.5)
attention.eval()
print(attention(queries, keys, values, valid_lens))
