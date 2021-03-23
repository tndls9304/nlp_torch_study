import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class Encoder(nn.Module):
    def __init__(self, config):
        # input_dim == vocab_size of src
        super().__init__()
        input_dim = config['input_dim']
        emb_dim = config['enc_emb_dim']
        self.enc_hid_dim = config['enc_hid_dim']
        dec_hid_dim = config['dec_hid_dim']
        self.n_layers = config['enc_n_layers']
        dropout = config['enc_dropout']
        rnn_dropout = config['enc_rnn_dropout']
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, self.enc_hid_dim, self.n_layers, dropout=rnn_dropout, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.enc_hid_dim * 2, dec_hid_dim)

    def forward(self, src, input_lengths):  # src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))  # embedded = [src len, batch size, emb dim]
        packed_input = pack_padded_sequence(embedded, input_lengths.tolist(), batch_first=True, enforce_sorted=True)
        packed_output, hidden = self.rnn(packed_input)
        outputs, lengths = pad_packed_sequence(packed_output, batch_first=True)
        # torch.Size([2(n_layers) * 2(bidirectional), 64, 512])

        # hidden = [n layers * n directions, batch size, hid dim]
        batch_size = outputs.shape[0]
        hidden = hidden.view(self.n_layers, 2, batch_size, self.enc_hid_dim)

        # (n_layers, direction, batch, hidden_size)
        hid_cat = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=-1).squeeze()
        # (1, 1, batch, hidden_size) + (1, 1, batch, hidden_size) ->
        # (1, 1, batch, hidden_size * 2) -> (2, batch, hidden_size*2)
        hidden = torch.tanh(self.fc(hid_cat))
        # torch.Size([64, 512])

        # outputs=[src len, batch size, hid dim * n directions]
        # outputs are always from the top hidden layer

        # outputs = [src len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]
        return outputs, hidden
