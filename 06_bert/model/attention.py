import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, head_dim, pad_idx, dropout=0.1):
        super().__init__()
        self.head_dim = head_dim
        self.pad_idx = pad_idx
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-1, -2))

        scores = scores.mul_(1 / (self.head_dim**0.5))
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        attn_prob = self.softmax(scores)
        p_attn = self.attn_dropout(attn_prob)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config['bert_hidden_size']
        self.n_head = config['num_heads']
        self.head_dim = config['bert_hidden_size'] // config['num_heads']

        # query, key, value
        self.linear_layers = nn.ModuleList([nn.Linear(config['bert_hidden_size'], config['bert_hidden_size']) for _ in range(3)])
        self.output_linear = nn.Linear(self.n_head * self.head_dim, config['bert_hidden_size'])
        self.attention = Attention(self.head_dim, config['pad_idx'], config['dropout_attn'])

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query, key, value = [layer(x).view(batch_size, -1, self.n_head, self.head_dim).transpose(1, 2) for layer, x in zip(self.linear_layers, (query, key, value))]
        scores, attn = self.attention(query, key, value, mask=mask)
        x = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.head_dim)
        return self.output_linear(x)
