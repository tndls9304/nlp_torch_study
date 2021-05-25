import torch
import torch.nn as nn

from torch.autograd import Variable

from .ci_embedding import ContextIndependentEmbedding
from .bi_lm import BidirectionalLM


def get_mask(batch_idx):
    batch_size, seq_max_len = batch_idx.shape
    masks = [torch.LongTensor(batch_size, seq_max_len).fill_(0), [], []]

    for i, x_i in enumerate(batch_idx):
        for j in range(len(x_i)):
            masks[0][i][j] = 1
            if j + 1 < len(x_i):
                masks[1].append(i * seq_max_len + j)
            if j > 0:
                masks[2].append(i * seq_max_len + j)

    assert len(masks[1]) <= batch_size * seq_max_len
    assert len(masks[2]) <= batch_size * seq_max_len

    masks[1] = torch.LongTensor(masks[1])
    masks[2] = torch.LongTensor(masks[2])

    return masks


class ELMO(nn.Module):
    def __init__(self, config):
        super(ELMO, self).__init__()
        self.config = config
        self.use_cuda = config['use_cuda']
        self.output_dim = config['output_dim']

        self.ci_embedding = ContextIndependentEmbedding(config, False)
        # self.frozen_embedding = ContextIndependentEmbedding(config, True)

        self.encoder = BidirectionalLM(config)
        self.decoder = nn.Linear(2 * config['emb_dim'], self.output_dim)
        # self.weighted_sum = nn.Linear(3, 1, bias=False)

    def forward(self, batch):
        enc_output, token_embedding = self.get_encoded(batch)
        return self.decoder(enc_output), token_embedding

    def get_encoded(self, batch):
        batch_idx, batch_length = batch
        mask_package = get_mask(batch_idx)
        mask = Variable(mask_package[0]).cuda() if self.use_cuda else Variable(mask_package[0])
        batch_embedding = self.ci_embedding(batch_idx)
        # print(batch_embedding.shape)
        # [batch_size, seq_len, emb_dim]

        encoder_output = self.encoder(batch_embedding, mask)
        # print(encoder_output.shape)
        # [2, batch_size, seq_len, emb_dim]
        sz = encoder_output.size()
        token_embedding = torch.cat(
            [batch_embedding, batch_embedding], dim=2).view(1, sz[1], sz[2], sz[3])
        # print(token_embedding.shape)
        # [1, batch_size, seq_len, emb_dim]

        # encoder_output = torch.cat([token_embedding, encoder_output], dim=0)
        # print(encoder_output.shape)
        # [3, batch_size, seq_len, emb_dim]
        # encoder_output = encoder_output.reshape(encoder_output.shape[1], -1, encoder_output.shape[-1], 3)
        # [batch_size, seq_len, emb_dim, 3]
        # encoder_output = self.weighted_sum(encoder_output).squeeze(-1)
        # [batch_size, seq_len, emb_dim]

        return encoder_output, token_embedding
