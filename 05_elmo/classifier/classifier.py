import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from gensim.models import KeyedVectors, Doc2Vec


class BaseClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gru = nn.GRU(config['emb_dim'], config['emb_dim'], dropout=config['rnn_dropout'],
                          num_layers=3, bidirectional=True)
        self.dropout = nn.Dropout(config['dropout'])
        self.fc = nn.Linear(2*config['emb_dim'], config['output_dim'])

    def get_embedded_vector(self, batch):
        raise NotImplemented

    def forward(self, batch):
        embedded_batch, length = self.get_embedded_vector(batch)
        length = length.to('cpu').tolist()
        packed = pack_padded_sequence(embedded_batch, length, batch_first=True, enforce_sorted=False)

        packed_output, _ = self.gru(packed)

        seq_unpacked, lens_unpacked = pad_packed_sequence(packed_output, batch_first=True)
        seq_unpacked = self.dropout(seq_unpacked)
        seq_unpacked = seq_unpacked[:, -1, :].squeeze()
        seq_unpacked = self.fc(seq_unpacked)

        return seq_unpacked


class RandomEmbeddingCLS(BaseClassifier):
    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config['input_dim'], config['emb_dim'])

    def get_embedded_vector(self, batch):
        batch_token, length = batch
        return self.embedding(batch_token), length


class PreTrainedW2VEmbeddingCLS(BaseClassifier):
    def __init__(self, config):
        super().__init__(config)
        w2v = KeyedVectors.load_word2vec_format(config['word2vec_path'])
        weights = torch.FloatTensor(w2v.vectors)
        self.embedding = nn.Embedding.from_pretrained(weights, False)

    def get_embedded_vector(self, batch):
        batch_token, length = batch
        return self.embedding(batch_token), length
