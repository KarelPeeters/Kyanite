import torch
from torch import nn


class Flip(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.flip(x, [self.dim])
