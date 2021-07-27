import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

from gensim.models import KeyedVectors


class PreTrainedEmbedding(nn.Module):
    # def __init__(self, src_vocab, w2v, freeze=False):
    # self.w2v = w2v
    # self.src_vocab = src_vocab
    def __init__(self, config, freeze=False):
        super().__init__()
        # weights = torch.FloatTensor(self.match_vocab())
        # self.embedding = nn.Embedding.from_pretrained(weights, freeze=freeze)
        self.embedding = nn.Embedding(config['src_vocab_size'], config['embed_dim'])

    def forward(self, batch):
        embedding = self.embedding(batch)
        return embedding

    def match_vocab(self):
        weights = self.w2v.vectors
        new_weights = deepcopy(weights)
        for i, word in enumerate(self.src_vocab.itos):
            if word in self.src_vocab.stoi:
                new_weights[i, :] = weights[self.src_vocab.stoi[word], :]
            else:
                new_weights[i, :] = np.random.normal(0, 0.01, size=(1, 200))
        return new_weights
