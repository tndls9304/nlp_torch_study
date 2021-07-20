import torch
import torch.nn as nn


class CRFLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tag_vocab_size = self.config['tag_vocab_size']
        self.transition = nn.Parameter(torch.randn(self.tag_vocab_size, self.tag_vocab_size))

    def forward(self, tags, mask, emit_score):
        batch_size, seq_len = tags.shape
        score = torch.gather(emit_score, dim=2, index=tags.unsqueeze(dim=2)).squeeze(dim=2)
        score[:, 1:] += self.transition[tags[:, :-1], tags[:, 1:]]
        total_score = (score * mask.type(torch.float)).sum(dim=1)

        d = torch.unsqueeze(emit_score[:, 0], dim=1)
        for i in range(1, seq_len):
            n_unfinished = mask[:, i].sum()
            d_uf = d[:, n_unfinished]
            emit_and_transition = emit_score[:n_unfinished, i].unsqueeze(1) + self.transition
            log_sum = d_uf.transpose(1, 2) + emit_and_transition
            max_v = log_sum.max(dim=1)[0].unsqueeze(1)
            log_sum = log_sum - max_v
            d_uf = max_v + torch.logsumexp(log_sum, dim=1).unsqueeze(1)
            d = torch.cat((d_uf, d[n_unfinished:]), dim=0)

        d = d.squeeze(dim=1)
        max_d = d.max(dim=-1)[0]
        d = max_d + torch.logsumexp(d - max_d.unsqueeze(1), dim=1)
        llk = total_score - d
        return -llk


