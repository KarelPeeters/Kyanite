from typing import List

import numpy as np
import torch

from lib.games import Game
from lib.util import DEVICE, prod


class Position:
    def __init__(self, game: Game, scalar_names: List[str], data: bytes):
        data = Taker(data)

        scalar_array = np.frombuffer(data.take(len(scalar_names) * 4), dtype=np.float32)
        scalars = {n: v for n, v in zip(scalar_names, scalar_array)}

        self.game_id = int(scalars.pop("game_id"))
        self.pos_index = int(scalars.pop("pos_index"))
        self.game_length = int(scalars.pop("game_length"))
        self.zero_visits = int(scalars.pop("zero_visits"))
        self.available_mv_count = int(scalars.pop("available_mv_count"))

        played_mv_float = scalars.pop("played_mv", None)
        self.played_mv = int(played_mv_float) if played_mv_float is not None else None

        self.kdl_policy = float(scalars.pop("kdl_policy"))

        self.final_v = float(scalars.pop("final_v"))
        self.zero_v = float(scalars.pop("zero_v"))
        self.net_v = float(scalars.pop("net_v"))

        self.final_wdl = np.array([scalars.pop("final_wdl_w"), scalars.pop("final_wdl_d"), scalars.pop("final_wdl_l")])
        self.zero_wdl = np.array([scalars.pop("zero_wdl_w"), scalars.pop("zero_wdl_d"), scalars.pop("zero_wdl_l")])
        self.net_wdl = np.array([scalars.pop("net_wdl_w"), scalars.pop("net_wdl_d"), scalars.pop("net_wdl_l")])

        self.final_moves_left = float(scalars.pop("final_moves_left", self.game_length - self.pos_index))
        self.zero_moves_left = float(scalars.pop("zero_moves_left", np.nan))
        self.net_moves_left = float(scalars.pop("net_moves_left", np.nan))

        if len(scalars):
            print(f"Leftover scalars: {list(scalars.keys())}")

        bool_count = prod(game.input_bool_shape)
        bit_buffer = np.frombuffer(data.take((bool_count + 7) // 8), dtype=np.uint8)
        bool_buffer = np.unpackbits(bit_buffer, bitorder="little")
        self.input_bools = bool_buffer[:bool_count]
        self.input_scalars = np.frombuffer(data.take(game.input_scalar_channels * 4), dtype=np.float32)

        self.policy_indices = np.frombuffer(data.take(self.available_mv_count * 4), dtype=np.int32)
        self.policy_values = np.frombuffer(data.take(self.available_mv_count * 4), dtype=np.float32)

        data.finish()


class PositionBatch:
    def __init__(self, game: Game, positions: List[Position], pin_memory: bool):
        self.max_available_moves = max(p.available_mv_count for p in positions)

        input_full = torch.empty(len(positions), *game.full_input_shape, pin_memory=pin_memory)
        all_wdls = torch.empty(len(positions), 3 * 3, pin_memory=pin_memory)
        all_values = torch.empty(len(positions), 3, pin_memory=pin_memory)
        all_moves_left = torch.empty(len(positions), 3, pin_memory=pin_memory)

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
            all_values[i, 0] = p.final_v
            all_values[i, 1] = p.zero_v
            all_values[i, 2] = p.net_v
            all_moves_left[i, 0] = p.final_moves_left
            all_moves_left[i, 1] = p.zero_moves_left
            all_moves_left[i, 2] = p.net_moves_left

            policy_indices[i, :p.available_mv_count] = torch.from_numpy(p.policy_indices.copy())
            policy_values[i, :p.available_mv_count] = torch.from_numpy(p.policy_values.copy())

        self.input_full = input_full.to(DEVICE)
        self.policy_indices = policy_indices.to(DEVICE)
        self.policy_values = policy_values.to(DEVICE)

        self.all_wdls = all_wdls.to(DEVICE)
        self.all_values = all_values.to(DEVICE)
        self.all_moves_left = all_moves_left.to(DEVICE)

        self.wdl_final = self.all_wdls[:, 0:3]
        self.wdl_zero = self.all_wdls[:, 3:6]
        self.wdl_net = self.all_wdls[:, 6:9]
        self.v_final = self.all_values[:, 0]
        self.v_zero = self.all_values[:, 1]
        self.v_net = self.all_values[:, 2]
        self.moves_left_final = self.all_moves_left[:, 0]
        self.moves_left_zero = self.all_moves_left[:, 1]
        self.moves_left_net = self.all_moves_left[:, 2]

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
