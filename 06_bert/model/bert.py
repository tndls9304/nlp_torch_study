import torch.nn as nn

from .transformer import Transformer
from .embeddings import BERTEmbedding


class BERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.b_embedding = BERTEmbedding(config)
        self.transformers = nn.ModuleList([Transformer(config) for _ in range(config['transformer_layer'])])

    def forward(self, input_batch, segment, position):
        mask = (input_batch != self.config['pad_idx']).unsqueeze(1).repeat(1, input_batch.size(1), 1).unsqueeze(1)

        hidden = self.b_embedding(input_batch, segment, position)

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
