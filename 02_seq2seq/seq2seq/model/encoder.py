import torch.nn as nn

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class Encoder(nn.Module):
    def __init__(self, config):
        # input_dim == vocab_size of src
        super().__init__()
        self.input_dim = config['input_dim']
        self.emb_dim = config['enc_emb_dim']
        self.hid_dim = config['enc_hid_dim']
        self.n_layers = config['enc_n_layers']
        self.embedding = nn.Embedding(self.input_dim, self.emb_dim)
        self.rnn = nn.LSTM(self.emb_dim, self.hid_dim, self.n_layers, dropout=config['enc_rnn_dropout'])
        self.dropout = nn.Dropout(config['enc_dropout'])

    def forward(self, src, input_lengths):  # src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))  # embedded = [src len, batch size, emb dim]
        # embedded = embedded.permute(1, 0, 2)
        # print(embedded.shape)
        packed_input = pack_padded_sequence(embedded, input_lengths.tolist(), batch_first=True)
        packed_output, (hidden, cell) = self.rnn(packed_input)  # outputs=[src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        # outputs are always from the top hidden layer
        return hidden, cell

