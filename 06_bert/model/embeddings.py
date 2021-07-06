import math
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):

    def __init__(self, embedding_dim, max_len):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embedding_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class BERTEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.t_embedding = nn.Embedding(config['vocab_size']+1, config['bert_hidden_size'])
        self.p_embedding = PositionalEmbedding(config['bert_hidden_size'], config['max_len'])
        self.s_embedding = nn.Embedding(3, config['bert_hidden_size'])
        self.dropout = nn.Dropout(config['dropout_embed'])

    def forward(self, input_batch, segment):
        te = self.t_embedding(input_batch)
        pe = self.p_embedding(input_batch)
        se = self.s_embedding(segment)
        output = te + pe + se
        return self.dropout(output)
