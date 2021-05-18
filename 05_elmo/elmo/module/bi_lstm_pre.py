import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BidirectionalLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_layers = config['enc_n_layers']
        self.fLSTM = nn.LSTM(config['emb_dim'], config['enc_hid_dim'], self.n_layers, dropout=config['rnn_dropout'], batch_first=True)
        self.bLSTM = nn.LSTM(config['emb_dim'], config['enc_hid_dim'], self.n_layers, dropout=config['rnn_dropout'], batch_first=True)
        self.dropout = nn.Dropout(config['dropout'])
        self.fc = nn.Linear(config['enc_hid_dim'], config['emb_dim'])

    def forward(self, emb_src, input_lengths):  # src = [batch_size, seq_len, pre_embedded]
        packed_input = pack_padded_sequence(emb_src, input_lengths.tolist(), batch_first=True, enforce_sorted=True)
        packed_output, (hidden, state) = self.biLSTM(packed_input)
        outputs, lengths = pad_packed_sequence(packed_output, batch_first=True)
        # outputs: [batch_size, seq_len, hidden_dim * directions]

        batch_size = outputs.shape[0]

        outputs = outputs.reshape(batch_size, -1, self.config['enc_hid_dim'], 2)
        # outputs: [batch_size, seq_len, hidden_dim, directions]

        forward_hidden = outputs[:, :, :, 0]
        backward_hidden = outputs[:, :, :, 1]
        # f/b hidden [batch_size, seq_len, hidden_dim]

        forward_hidden = self.fc(self.dropout(forward_hidden))
        backward_hidden = self.fc(self.dropout(backward_hidden))
        # f/b hidden [batch_size, seq_len, emb_dim]

        elmo_embedding = torch.cat((forward_hidden, emb_src, backward_hidden), dim=-1)
        # elmo_embedding [batch_size, seq_len, 3 * emb_dim]
        # print(elmo_embedding.shape)
        assert elmo_embedding.shape[-1] == self.config['elmo_embedding_size']

        return elmo_embedding
