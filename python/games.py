from dataclasses import dataclass

from util import prod


@dataclass
class Game:
    name: str
    board_size: int
    input_channels: int
    policy_channels: int

    @property
    def input_shape(self):
        return self.input_channels, self.board_size, self.board_size

    @property
    def policy_shape(self):
        return self.policy_channels, self.board_size, self.board_size

    @property
    def input_size(self):
        return prod(self.input_shape)

    @property
    def policy_size(self):
        return prod(self.policy_shape)

    @property
    def data_width(self):
        return 3 + 3 + 2 * self.policy_size + self.input_size


# TODO add repetition and n-move rule counters to board inputs?
# TODO try separating en passant for different colors

GAMES = [
    Game(
        name="ataxx",
        board_size=7,
        input_channels=3,
        policy_channels=17,
    ),
    Game(
        name="chess",
        board_size=8,
        input_channels=2 + 6 * 2 + 4 + 1,
        policy_channels=7 * 8 + 8 + 3 * 3,
    ),
    Game(
        name="sttt",
        board_size=9,
        input_channels=3,
        policy_channels=1,
    )
]


def find_game(name: str):
    for game in GAMES:
        if game.name == name:
            return game
    raise KeyError("Game '{}' not found", name)
