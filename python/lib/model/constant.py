import torch
from torch import nn

from lib.games import Game
from lib.model.model import GameNetwork


class ConstantNetwork(GameNetwork):
    def __init__(self, game: Game):
        super().__init__()

        self.value = nn.Parameter(torch.zeros(1))
        self.wdl = nn.Parameter(torch.zeros(1, 3))
        self.policy = nn.Parameter(torch.zeros(1, *game.policy_shape))

    def forward(self, input):
        batch_size = input.shape[0]

        value = self.value.expand(batch_size)
        wdl = self.wdl.expand(batch_size, -1)
        policy = self.policy.expand(batch_size, -1, -1, -1)

        return value, wdl, policy
