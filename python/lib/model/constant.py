import torch
from torch import nn

from lib.games import Game


class ConstantNetwork(nn.Module):
    def __init__(self, game: Game):
        super().__init__()

        self.scalars = nn.Parameter(torch.randn(1, 5))
        self.policy = nn.Parameter(torch.randn(1, *game.policy_shape))

    def forward(self, input):
        batch_size = input.shape[0]

        scalars = self.value.expand(batch_size, -1)
        policy = self.policy.expand(batch_size, -1, -1, -1)

        return scalars, policy
