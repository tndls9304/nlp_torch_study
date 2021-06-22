import torch.nn as nn


class PositionWiseFF(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_second = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, input_batch):
        hidden = self.linear_first(input_batch)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        return self.linear_second(hidden)
