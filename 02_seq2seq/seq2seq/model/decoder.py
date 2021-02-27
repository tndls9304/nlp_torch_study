import torch.nn as nn

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_dim = config['output_dim']  # output_dim == vocab_size of trg
        self.hid_dim = config['dec_hid_dim']
        self.n_layers = config['dec_n_layers']
        self.emb_dim = config['dec_emb_dim']
        self.embedding = nn.Embedding(self.output_dim, self.emb_dim)
        self.rnn = nn.LSTM(self.emb_dim, self.hid_dim, self.n_layers, dropout=config['dec_rnn_dropout'])
        self.fc_out = nn.Linear(self.hid_dim, self.output_dim)
        self.dropout = nn.Dropout(config['dec_dropout'])

    def forward(self, dec_input, hidden, cell):
        # input = [batch size]
        # print('di', dec_input.shape)

        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        dec_input = dec_input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(dec_input))  # embedded = [1, batch size, emb dim]
        # print(embedded.shape)
        # packed_input = pack_padded_sequence(embedded, input_lengths.tolist(), batch_first=True)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # print(output.shape)

        # output, output_lengths = pad_packed_sequence(packed_output)
        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
        # print('o', output.shape)
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch size, output dim]
        return prediction, hidden, cell
