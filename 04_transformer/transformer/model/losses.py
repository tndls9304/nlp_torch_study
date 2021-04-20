import torch.nn as nn
import torch.nn.functional as F


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean', ignore_index=-100):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        n = pred.size()[-1]
        log_pred = F.log_softmax(pred, dim=-1)
        loss = reduce_loss(-log_pred.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_pred, target, reduction=self.reduction, ignore_index=self.ignore_index)
        return linear_combination(loss / n, nll, self.epsilon)
