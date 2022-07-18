from torch import nn

from lib.games import Game
from lib.util import prod


class DenseNetwork(nn.Module):
    def __init__(self, game: Game, depth: int, size: int, res: bool):
        super().__init__()
        assert depth >= 1, "Need at least one hidden layer"

        self.policy_shape = game.policy_shape

        layers = [
            nn.Flatten(),
            nn.Linear(prod(game.full_input_shape), size),
            *[DenseBlock(size, res) for _ in range(depth)],
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Linear(size, 5 + game.policy_size)
        ]

        self.seq = nn.Sequential(*layers)

    def forward(self, input):
        output = self.seq(input)

        scalars = output[:, :5]
        policy = output[:, 5:].view(-1, *self.policy_shape)

        return scalars, policy


class DenseBlock(nn.Module):
    def __init__(self, size: int, res: bool):
        super().__init__()
        self.res = res
        self.seq = nn.Sequential(
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Linear(size, size)
        )

    def forward(self, x):
        y = self.seq(x)
        if self.res:
            return x + y
        else:
            return y
