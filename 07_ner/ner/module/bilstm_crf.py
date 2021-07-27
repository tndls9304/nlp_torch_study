import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF

from .embedding import PreTrainedEmbedding


class Highway(nn.Module):
    def __init__(self, size, num_layers, f):

        super(Highway, self).__init__()

        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = f

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x


class BiLSTMCRF(nn.Module):
    def __init__(self, config, w2v):
        super().__init__()
        self.config = config
        self.dropout_rate = config['dropout']
        self.embed_dim = config['embed_dim']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.vocab_size = config['src_vocab_size']
        self.tag_size = config['trg_vocab_size']

        # self.embedding = PreTrainedEmbedding(config['src_vocab'], w2v)
        self.embedding = PreTrainedEmbedding(config)
        self.highway = Highway(config['embed_dim'], config['highway_n_layer'], torch.relu)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.encoder = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_size, bidirectional=True,
                               num_layers=self.num_layers, dropout=0.05)

        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.hidden2score = nn.Linear(2 * self.hidden_size, self.tag_size)
        self.CRF = CRF(self.tag_size, batch_first=True)

    def forward(self, batch_src, batch_trg):
        sentence, sentence_length = batch_src
        tags = batch_trg
        mask = (sentence != self.config['pad_idx']).to(self.config['device'])
        sentence = self.embedding(sentence)
        # sentence = self.highway(self.embedding(sentence))
        sentence = self.layer_norm(sentence)
        # print(sentence.shape)
        emit_score = self.encode(sentence, sentence_length)
        loss = self.CRF(emit_score, tags, mask=mask, reduction='mean')
        return emit_score, loss, mask  # (b, l, t)

    def encode(self, sentence, length):
        length = length.to('cpu').tolist()
        padded_sent = pack_padded_sequence(sentence, length, batch_first=True)
        hidden, _ = self.encoder(padded_sent)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True)
        # hidden = self.layer_norm(hidden)
        emit_score = self.hidden2score(hidden)
        return self.dropout(emit_score)
