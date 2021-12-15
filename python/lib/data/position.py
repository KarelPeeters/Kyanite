from typing import List

import numpy as np
import torch

from lib.games import Game
from lib.util import DEVICE, prod

POSITION_INFO_SCALAR_COUNT = 6 + 4 * 3


# TODO properly parse column headers here instead of depending on indices
class Position:
    def __init__(self, game: Game, data: bytes):
        data = Taker(data)

        scalars = Taker(np.frombuffer(data.take(POSITION_INFO_SCALAR_COUNT * 4), dtype=np.float32))

        [self.game_id, self.pos_index, self.game_length, self.zero_visits, self.available_moves] = \
            scalars.take(5).astype(int)
        self.kdl_policy = scalars.take(1)

        self.final_v = scalars.take(1)
        self.final_wdl = scalars.take(3)
        self.zero_v = scalars.take(1)
        self.zero_wdl = scalars.take(3)
        self.net_v = scalars.take(1)
        self.net_wdl = scalars.take(3)

        scalars.finish()

        bool_count = prod(game.input_bool_shape)
        bit_buffer = np.frombuffer(data.take((bool_count + 7) // 8), dtype=np.uint8)
        bool_buffer = np.unpackbits(bit_buffer, bitorder="little")
        self.input_bools = bool_buffer[:bool_count]
        self.input_scalars = np.frombuffer(data.take(game.input_scalar_channels * 4), dtype=np.float32)

        self.policy_indices = np.frombuffer(data.take(self.available_moves * 4), dtype=np.int32)
        self.policy_values = np.frombuffer(data.take(self.available_moves * 4), dtype=np.float32)

        data.finish()


class PositionBatch:
    def __init__(self, game: Game, positions: List[Position], pin_memory: bool):
        self.max_available_moves = max(p.available_moves for p in positions)

        input_full = torch.empty(len(positions), *game.full_input_shape, pin_memory=pin_memory)
        all_wdls = torch.empty(len(positions), 3 * 3, pin_memory=pin_memory)

        policy_indices = torch.zeros(len(positions), self.max_available_moves, dtype=torch.int64, pin_memory=pin_memory)
        policy_values = torch.empty(len(positions), self.max_available_moves, pin_memory=pin_memory)
        policy_values.fill_(-1)

        for i, p in enumerate(positions):
            input_full[i, :game.input_scalar_channels, :, :] = torch.from_numpy(p.input_scalars) \
                .view(-1, 1, 1).expand(*game.input_scalar_shape)
            input_full[i, game.input_scalar_channels:, :, :] = torch.from_numpy(p.input_bools) \
                .view(*game.input_bool_shape)

            all_wdls[i, 0:3] = torch.from_numpy(p.final_wdl)
            all_wdls[i, 3:6] = torch.from_numpy(p.zero_wdl)
            all_wdls[i, 6:9] = torch.from_numpy(p.net_wdl)
            policy_indices[i, :p.available_moves] = torch.from_numpy(p.policy_indices.copy())
            policy_values[i, :p.available_moves] = torch.from_numpy(p.policy_values.copy())

        self.input_full = input_full.to(DEVICE)
        self.policy_indices = policy_indices.to(DEVICE)
        self.policy_values = policy_values.to(DEVICE)

        self.all_wdls = all_wdls.to(DEVICE)
        self.wdl_final = self.all_wdls[:, 0:3]
        self.wdl_zero = self.all_wdls[:, 3:6]
        self.wdl_net = self.all_wdls[:, 6:9]

    # TODO use the proper value column for this instead
    def value_zero(self):
        return self.wdl_zero[:, 0] - self.wdl_zero[:, 2]

    def value_final(self):
        return self.wdl_final[:, 0] - self.wdl_final[:, 2]

    def __len__(self):
        return len(self.input_full)


class Taker:
    def __init__(self, inner):
        self.inner = inner
        self.next = 0

    def take(self, n: int):
        self.next += n
        return self.inner[self.next - n:self.next]

    def finish(self):
        assert self.next == len(self.inner), f"Only read {self.next}/{len(self.inner)} bytes"
