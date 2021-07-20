import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF

from .embedding import PreTrainedEmbedding


class BiLSTMCRF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout_rate = self.config['dropout']
        self.embed_dim = self.config['embed_dim']
        self.hidden_size = self.config['hidden_size']
        self.num_layers = self.config['num_layers']
        self.vocab_size = self.config['src_vocab_size']
        self.tag_size = self.config['trg_vocab_size']

        self.embedding = PreTrainedEmbedding(config)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.encoder = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_size, bidirectional=True,
                               num_layers=self.num_layers)

        self.CRF = CRF(self.tag_size, batch_first=True)
        self.hidden2score = nn.Linear(2 * self.hidden_size, self.tag_size)

    def forward(self, batch_src, batch_trg):
        sentence, sentence_length = batch_src
        tags = batch_trg
        mask = (sentence != self.config['pad_idx']).to(self.config['device'])
        sentence = self.embedding(sentence)
        # print(sentence.shape)
        emit_score = self.encode(sentence, sentence_length)
        loss = self.CRF(emit_score, tags, mask=mask, reduction='mean')
        return emit_score, loss, mask  # (b, l, t)

    def encode(self, sentence, length):
        length = length.to('cpu').tolist()
        padded_sent = pack_padded_sequence(sentence, length, batch_first=True)
        hidden, _ = self.encoder(padded_sent)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True)
        emit_score = self.hidden2score(hidden)
        return self.dropout(emit_score)

