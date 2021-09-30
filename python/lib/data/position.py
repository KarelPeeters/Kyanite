from math import prod
from typing import List

import numpy as np
import torch

from lib.games import Game, SCALAR_COUNT


class Position:
    def __init__(self, game: Game, data: bytes):
        data = Taker(data)

        scalars = Taker(np.frombuffer(data.take(SCALAR_COUNT * 4), dtype=np.float32))

        [self.game_id, self.pos_index, self.game_length, self.zero_visits, self.available_moves] = \
            scalars.take(5).astype(int)
        [self.kdl_wdl, self.kdl_policy] = scalars.take(2)
        self.wdls = scalars.take(3 * 3)
        scalars.finish()

        self.input_bools = np.unpackbits(
            np.frombuffer(data.take((prod(game.input_bool_shape) + 7) // 8), dtype=np.uint8))
        self.input_scalars = np.frombuffer(data.take(game.input_scalar_channels * 4), dtype=np.float32)

        self.policy_indices = np.frombuffer(data.take(self.available_moves * 4), dtype=np.int32)
        self.policy_values = np.frombuffer(data.take(self.available_moves * 4), dtype=np.float32)

        data.finish()


class PositionBatch:
    def __init__(self, positions: List[Position]):
        max_available_moves = max(p.available_moves for p in positions)

        self.wdls = torch.empty(len(positions), 3 * 3)
        self.available_moves = torch.empty(len(positions), dtype=torch.int32)
        self.policy_indices = torch.empty(len(positions), max_available_moves, dtype=torch.int32)
        self.policy_values = torch.empty(len(positions), max_available_moves)

        for i, p in enumerate(positions):
            self.wdls[i, :] = torch.from_numpy(p.wdls.copy())
            self.available_moves[i] = p.available_moves
            self.policy_indices[i, :p.available_moves] = torch.from_numpy(p.policy_indices.copy())
            self.policy_values[i, :p.available_moves] = torch.from_numpy(p.policy_values.copy())


class Taker:
    def __init__(self, inner):
        self.inner = inner
        self.next = 0

    def take(self, n: int):
        self.next += n
        return self.inner[self.next - n:self.next]

    def finish(self):
        assert self.next == len(self.inner), f"Only read {self.next}/{len(self.inner)} bytes"
