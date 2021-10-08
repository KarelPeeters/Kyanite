from numpy import product
from torch import nn

from lib.games import Game


class SimpleNetwork(nn.Module):
    def __init__(self, game: Game, depth: int, size: int, bn: bool):
        super().__init__()
        assert depth >= 1, "Need at least one hidden layer"

        self.policy_shape = game.policy_shape

        layers = [
            nn.Flatten(),
            nn.Linear(product(game.full_input_shape), size)
        ]

        if bn:
            layers.append(nn.BatchNorm1d(size))
        layers.append(nn.ReLU())

        for _ in range(depth - 1):
            if bn:
                layers.append(nn.Linear(size, size))
                layers.append(nn.BatchNorm1d(size))
                layers.append(nn.ReLU())

        layers.append(nn.Linear(size, 4 + product(game.policy_shape)))

        self.seq = nn.Sequential(*layers)

    def forward(self, input):
        output = self.seq(input)

        value = output[:, 0]
        wdl = output[:, 1:4]
        policy = output[:, 4:].view(-1, *self.policy_shape)

        return value, wdl, policy
