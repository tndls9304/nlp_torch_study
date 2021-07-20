import torch
import torch.nn as nn

from gensim.models import KeyedVectors


class PreTrainedEmbedding(nn.Module):
    def __init__(self, config, freeze=True):
        super().__init__()
        # w2v = KeyedVectors.load_word2vec_format(config['word2vec_path'])
        # weights = torch.FloatTensor(w2v.vectors)
        # self.embedding = nn.Embedding.from_pretrained(weights, freeze=freeze)
        self.embedding = nn.Embedding(config['src_vocab_size'], config['embed_dim'])

    def forward(self, batch):
        embedding = self.embedding(batch)
        return embedding
