import re
from dataclasses import dataclass, field
from typing import Tuple, Optional, Callable

import numpy as np

from lib.chess_mapping.chess_mapping import CHESS_FLAT_TO_MOVE_INPUT


@dataclass
class Game:
    name: str

    board_size: int
    input_bool_channels: int
    input_scalar_channels: int
    input_mv_channels: Optional[int]

    policy_shape: Tuple[int, ...]
    policy_conv_channels: Optional[int]

    estimate_moves_per_game: float

    input_bool_shape: Tuple[int, int, int] = field(init=False)
    input_scalar_shape: Tuple[int, int, int] = field(init=False)
    input_mv_shape: Optional[Tuple[int, int, int]] = field(init=False)

    full_input_channels: int = field(init=False)
    full_input_shape: Tuple[int, int, int] = field(init=False)

    encode_mv: Optional[Callable[[int], np.array]]

    def __post_init__(self):
        self.input_bool_shape = (self.input_bool_channels, self.board_size, self.board_size)
        self.input_scalar_shape = (self.input_scalar_channels, self.board_size, self.board_size)

        self.full_input_channels = self.input_bool_channels + self.input_scalar_channels
        self.full_input_shape = (self.full_input_channels, self.board_size, self.board_size)

        if self.input_mv_channels is not None:
            self.input_mv_shape = (self.input_mv_channels, self.board_size, self.board_size)
        else:
            self.input_mv_shape = None

    @classmethod
    def find(cls, name: str):
        if name == "ataxx":
            name = "ataxx-7"

        if name in GAMES:
            game = GAMES[name]
            assert game.name == name
            return game

        game = None
        m = re.match(r"ataxx-(\d+)", name)
        if m:
            game = _ataxx_game(int(m.group(1)))
        m = re.match(r"chess-hist-(\d+)", name)
        if m:
            game = _chess_hist_game(int(m.group(1)))

        if game is None:
            raise KeyError("Game '{}' not found", name)

        GAMES[name] = game
        assert game.name == name
        return game


def _ataxx_game(size: int):
    assert 2 <= size <= 8
    return Game(
        name=f"ataxx-{size}",
        board_size=size,
        input_bool_channels=3,
        input_scalar_channels=0,
        input_mv_channels=None,
        policy_shape=(17, size, size),
        policy_conv_channels=17,
        # estimated from fully random games
        estimate_moves_per_game=[0, 4, 19, 51, 106, 183, 275][size - 2],
        encode_mv=None,
    )


def _chess_hist_game(length: int):
    assert length >= 0
    chess = GAMES["chess"]
    return Game(
        name=f"chess-hist-{length}",
        board_size=chess.board_size,
        input_bool_channels=1 + (length + 1) * (2*6),
        input_scalar_channels=7 + (length + 1),
        input_mv_channels=chess.input_mv_channels,
        policy_shape=chess.policy_shape,
        policy_conv_channels=chess.policy_conv_channels,
        estimate_moves_per_game=chess.estimate_moves_per_game,
        encode_mv=chess.encode_mv,
    )


def encode_chess_move(mv: int) -> np.array:
    encoded = CHESS_FLAT_TO_MOVE_INPUT[mv, :]
    result = np.zeros((8, 8, 8), dtype=np.uint8)

    # from, to
    result[0, :, :].reshape(-1)[encoded[0]] = 1
    result[1, :, :].reshape(-1)[encoded[1]] = 1

    # other boolean planes
    result[2:, :, :] = encoded[2:, None, None]
    return result


def encode_ttt_move(mv: int) -> np.array:
    result = np.zeros((1, 3, 3))
    result.reshape(-1)[mv] = 1
    return result


GAMES = {
    "chess": Game(
        name="chess",
        board_size=8,
        input_bool_channels=13,
        input_scalar_channels=8,
        input_mv_channels=8,
        policy_shape=(1880,),
        policy_conv_channels=73,
        estimate_moves_per_game=150,
        encode_mv=encode_chess_move,
    ),
    "sttt": Game(
        name="sttt",
        board_size=9,
        input_bool_channels=3,
        input_scalar_channels=0,
        input_mv_channels=None,
        policy_shape=(1, 9, 9),
        policy_conv_channels=1,
        estimate_moves_per_game=40,
        encode_mv=None,
    ),
    "ttt": Game(
        name="ttt",
        board_size=3,
        input_bool_channels=2,
        input_scalar_channels=0,
        input_mv_channels=1,
        policy_shape=(1, 3, 3),
        policy_conv_channels=1,
        estimate_moves_per_game=5,
        encode_mv=encode_ttt_move,
    ),
}
