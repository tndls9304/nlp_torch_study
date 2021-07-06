import torch.nn as nn


from .attention import MultiHeadAttention
from .pff import PositionWiseFF
from .residual import ResidualConnect


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = MultiHeadAttention(config)
        self.pff = PositionWiseFF(config['bert_hidden_size'], config['pff_hidden_dim'], config['dropout_pff'])
        self.first_residual = ResidualConnect(config['bert_hidden_size'], config['dropout_norm'])
        self.second_residual = ResidualConnect(config['bert_hidden_size'], config['dropout_norm'])
        self.dropout = nn.Dropout(config['dropout_trans'])

    def forward(self, input_batch, mask):
        hidden = self.first_residual(input_batch, lambda h: self.attention(h, h, h, mask=mask))
        hidden = self.second_residual(hidden, self.pff)
        return self.dropout(hidden)
