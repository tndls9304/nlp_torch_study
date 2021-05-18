import torch
import torch.nn as nn

from gensim.models import KeyedVectors

from .highway import Highway


class ContextIndependentEmbedding(nn.Module):
    def __init__(self, config, freeze=True):
        super().__init__()
        self.config = config
        w2v = KeyedVectors.load_word2vec_format(config['word2vec_path'])
        weights = torch.FloatTensor(w2v.vectors)
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=freeze)
        self.highway = Highway(config['emb_dim'], config['highway_n_layer'], torch.relu)

    def forward(self, batch):
        embedding = self.embedding(batch)
        return self.highway(embedding)
