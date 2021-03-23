import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, config, attention):
        super().__init__()
        self.output_dim = config['output_dim']
        emb_dim = config['dec_emb_dim']
        enc_hid_dim = config['enc_hid_dim']
        self.dec_hid_dim = config['dec_hid_dim']
        self.n_layers = config['dec_n_layers']
        dropout = config['dec_dropout']
        rnn_dropout = config['dec_rnn_dropout']
        self.attention = attention
        self.embedding = nn.Embedding(self.output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, self.dec_hid_dim, self.n_layers, batch_first=True, dropout=rnn_dropout)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + self.dec_hid_dim + emb_dim, self.output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputted, hidden, encoder_outputs):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        # hidden = [n layers * n directions, batch size, hid dim]

        # context = [n layers * n, batch size, hid dim]

        # input = [batch size]

        embedded = self.dropout(self.embedding(inputted))
        # embedded = [batch size, emb dim]
        a = self.attention(hidden, encoder_outputs)
        # a = [batch size, src len]
        a = a.unsqueeze(1)
        # a = [batch size, 1, src len]  torch.Size([64, 1, 10])
        # encoder_outputs = [batch size, src len, enc hid dim * 2] torch.Size([64, 10, 1024])

        weighted = torch.bmm(a, encoder_outputs)
        # weighted = [batch size, 1, enc hid dim * 2] [64, 1, 1024]

        embedded = embedded.unsqueeze(1)
        # [batch_size, 1, emb_dim] torch.Size([64, 1, 256])
        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input = [batch size, 1, (enc hid dim * 2) + emb dim] [batch_size, 1, (1024) + xxx]

        # can't use packed_padded_sequence
        output, hidden = self.rnn(rnn_input, hidden)
        last_hidden = hidden[-1, :, :].unsqueeze(1)
        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == last_hidden).all()

        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)
        # print(embedded.shape) torch.Size([64, 1, 256])
        # print(output.shape) torch.Size([64, 1, 512])
        # print(weighted.shape) torch.Size([64, 1, 1024])

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=-1))
        # prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0)
