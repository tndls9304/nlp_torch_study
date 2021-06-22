import torch.nn as nn


class ResidualConnect(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, layers):
        return x + self.dropout(layers(self.layer_norm(x)))
