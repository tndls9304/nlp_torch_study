import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
             else torch.arange(x.size(i)-1, -1, -1).long()
             for i in range(x.dim()))]


class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device, batch_first=False, dropout=0):
        super().__init__()
        self.layers = []
        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        for i in range(num_layers):
            if i == 0:
                input_dim = input_size
            else:
                input_dim = hidden_size
            lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=1, batch_first=batch_first,
                           dropout=dropout, bidirectional=False).to(self.device)
            self.layers.append(lstm)

    def forward(self, input_batch, batch_length):
        batch_size, seq_len = input_batch.shape[0], input_batch.shape[1]
        packed_batch = pack_padded_sequence(input_batch, batch_length, batch_first=True)
        stacked_output = torch.zeros((self.num_layers, batch_size, seq_len, self.hidden_size)).to(self.device)
        for i, layer in enumerate(self.layers):
            packed_batch, _ = layer(packed_batch)
            padded_batch, padded_length = pad_packed_sequence(packed_batch, batch_first=True)
            packed_batch = pack_padded_sequence(padded_batch, batch_length, batch_first=True)
            stacked_output[i] = padded_batch
        return stacked_output
