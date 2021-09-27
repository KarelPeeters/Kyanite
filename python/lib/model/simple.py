from torch import nn

from lib.games import Game


class SimpleNetwork(nn.Module):
    def __init__(self, game: Game, wdl: bool):
        super().__init__()

        self.value_size = 3 if wdl else 1
        self.policy_shape = game.policy_shape

        self.seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(game.input_size_history, 1024),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, game.policy_size + self.value_size),
        )

    def forward(self, input):
        output = self.seq(input)

        wdl = output[:, :self.value_size]
        policy = output[:, self.value_size:].view(-1, *self.policy_shape)

        print(wdl)

        return wdl, policy
