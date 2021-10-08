from numpy import product
from torch import nn

from lib.games import Game


class SimpleNetwork(nn.Module):
    def __init__(self, game: Game, depth: int, size: int):
        super().__init__()
        assert depth >= 1, "Need at least one hidden layer"

        self.policy_shape = game.policy_shape

        self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(product(game.full_input_shape), size),
            nn.ReLU(),
            *[x for x in [nn.Linear(size, size), nn.ReLU()] for _ in range(depth - 1)],
            nn.Linear(size, 4 + product(game.policy_shape)),
        )

        pass

    def forward(self, input):
        output = self.seq(input)

        value = output[:, 0]
        wdl = output[:, 1:4]
        policy = output[:, 4:].view(-1, *self.policy_shape)

        return value, wdl, policy
