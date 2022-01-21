from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class Game:
    name: str

    board_size: int
    input_bool_channels: int
    input_scalar_channels: int

    policy_shape: Tuple[int, ...]
    policy_conv_channels: Optional[int]

    estimate_moves_per_game: float

    input_bool_shape: Tuple[int, int, int] = field(init=False)
    input_scalar_shape: Tuple[int, int, int] = field(init=False)

    full_input_channels: int = field(init=False)
    full_input_shape: Tuple[int, int, int] = field(init=False)

    def __post_init__(self):
        self.input_bool_shape = (self.input_bool_channels, self.board_size, self.board_size)
        self.input_scalar_shape = (self.input_scalar_channels, self.board_size, self.board_size)

        self.full_input_channels = self.input_bool_channels + self.input_scalar_channels
        self.full_input_shape = (self.full_input_channels, self.board_size, self.board_size)

    @classmethod
    def find(cls, name: str):
        if name == "ataxx":
            name = "ataxx-7"

        for game in GAMES:
            if game.name == name:
                return game
        raise KeyError("Game '{}' not found", name)


def _ataxx_game(size: int):
    assert 2 <= size <= 8
    return Game(
        name=f"ataxx-{size}",
        board_size=size,
        input_bool_channels=3,
        input_scalar_channels=0,
        policy_shape=(17, size, size),
        policy_conv_channels=17,
        # estimated from fully random games
        estimate_moves_per_game=[0, 4, 19, 51, 106, 183, 275][size - 2],
    )


GAMES = [
    *(_ataxx_game(size) for size in range(2, 9)),
    Game(
        name="chess",
        board_size=8,
        input_bool_channels=13,
        input_scalar_channels=8,
        policy_shape=(1880,),
        policy_conv_channels=73,
        estimate_moves_per_game=100,
    ),
    Game(
        name="sttt",
        board_size=9,
        input_bool_channels=3,
        input_scalar_channels=0,
        policy_shape=(1, 9, 9),
        policy_conv_channels=1,
        estimate_moves_per_game=40,
    ),
    Game(
        name="ttt",
        board_size=3,
        input_bool_channels=2,
        input_scalar_channels=0,
        policy_shape=(1, 3, 3),
        policy_conv_channels=1,
        estimate_moves_per_game=5,
    )
]
