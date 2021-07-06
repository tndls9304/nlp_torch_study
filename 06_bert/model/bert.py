import torch.nn as nn

from .transformer import Transformer
from .embeddings import BERTEmbedding


class BERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.b_embedding = BERTEmbedding(config)
        self.transformers = nn.ModuleList([Transformer(config) for _ in range(config['transformer_layer'])])

    def forward(self, input_batch, segment):
        mask = (input_batch > self.config['pad_idx']).unsqueeze(1).repeat(1, input_batch.size(1), 1).unsqueeze(1)

        hidden = self.b_embedding(input_batch, segment)

        for transformer in self.transformers:
            hidden = transformer(hidden, mask)

        return hidden


class NextSentenceClassification(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))


class MLM(nn.Module):
    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class TrainableBERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = BERT(config)
        self.nsp_model = NextSentenceClassification(self.config['bert_hidden_size'])
        self.mlm_model = MLM(self.config['bert_hidden_size'], self.config['input_vocab_size'])

    def forward(self, x, segment):
        hidden = self.bert(x, segment)
        return self.nsp_model(hidden), self.mlm_model(hidden)
