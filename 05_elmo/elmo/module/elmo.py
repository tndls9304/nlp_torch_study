import torch
import torch.nn as nn

from torch.autograd import Variable

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
        self.output_dim = config['output_dim']

        self.encoder = BidirectionalLM(config)
        self.decoder = nn.Linear(config['emb_dim'], self.output_dim)

    def forward(self, batch):
        enc_output, _ = self.get_encoded(batch)
        # print(enc_output.shape)  (n_layer, batch_size, seq_len, elmo_dim*2)
        enc_forward_last = enc_output[-1, :, :, 0:self.config['elmo_embedding_size']].squeeze()
        enc_backward_last = enc_output[-1, :, :, self.config['elmo_embedding_size']:].squeeze()
        forward_pred = self.decoder(enc_forward_last)
        backward_pred = self.decoder(enc_backward_last)
        return forward_pred, backward_pred

    def get_encoded(self, batch):
        batch_idx, batch_length = batch
        mask_package = get_mask(batch_idx)
        mask = Variable(mask_package[0]).to(self.config['device'])

        encoder_output = self.encoder(batch_idx, mask)
        # print(encoder_output.shape)
        # [num_layer, batch_size, seq_len, emb_dim*2 (forward/backward]
        # (2, 16, n, 200*2)

        trained_ci_embedding = self.encoder.ci_embedding(batch_idx).unsqueeze(0)
        # print(batch_embedding.shape)
        # [1, batch_size, seq_len, emb_dim]

        return encoder_output, trained_ci_embedding
