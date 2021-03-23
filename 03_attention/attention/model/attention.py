import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        enc_hid_dim = config['enc_hid_dim']
        n_layers = config['enc_n_layers']
        dec_hid_dim = config['dec_hid_dim']
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim * n_layers)
        self.v = nn.Linear(dec_hid_dim * n_layers, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # print(hidden.shape) torch.Size([2, 64, 512])
        hidden = hidden[-1, :, :]
        hidden = hidden.reshape(batch_size, -1)

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # hidden = [batch size, src len, dec hid dim] [batch, len, hidden] [64, 10, 1024]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        attn_input = torch.cat((hidden, encoder_outputs), dim=2)
        # [batch, len, 1536 (1024 + 512)]

        energy = torch.tanh(self.attn(attn_input))
        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)
