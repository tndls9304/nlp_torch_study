import torch
import torch.nn as nn


class MaxOut(nn.Module):
    def __init__(self, input_dim, output_dim, pool_size):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.pool_size = pool_size
        self.maxout = nn.Linear(self.input_dim, self.output_dim * self.pool_size)

    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.output_dim
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.maxout(inputs)
        m, _ = out.view(*shape).max(max_dim)
        return m
