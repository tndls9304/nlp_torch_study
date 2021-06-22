import torch
import torch.nn as nn


class BERTEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.t_embedding = nn.Embedding(config['input_vocab_size'], config['bert_hidden_size'])
        self.p_embedding = nn.Embedding(config['max_len'], config['bert_hidden_size'])
        self.s_embedding = nn.Embedding(3, config['bert_hidden_size'])
        self.dropout = nn.Dropout(config['dropout_embed'])

    def forward(self, input_batch, segment, position):
        positions = torch.arange(input_batch.size(1), device=self.config['device'], dtype=torch.int).expand(input_batch.size()).contiguous() + 1
        positions_pad_mask = positions.eq(self.config['pad_idx'])
        positions.masked_fill_(positions_pad_mask, 0)
        output = self.t_embedding(input_batch) + self.p_embedding(position) + self.s_embedding(segment)
        return self.dropout(output)
