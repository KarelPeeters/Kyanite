import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from lib.data.taker import Taker
from lib.games import Game
from lib.util import DEVICE, prod, map_none, map_none_or


@dataclass
class Simulation:
    index: int
    start_file_pi: int
    move_count: int
    includes_final: bool

    @property
    def position_count(self):
        return self.move_count + self.includes_final

    @property
    def end_file_pi(self):
        """ The end position index, exclusive """
        return self.start_file_pi + self.position_count

    @property
    def file_pis(self):
        return range(self.start_file_pi, self.end_file_pi)


class Position:
    def __init__(
            self,
            game: Game,
            file_pi: int,
            includes_final: bool,
            scalar_names: List[str],
            data: bytes,
            final_position: Optional['Position']
    ):
        self.game = game
        data = Taker(data)

        scalar_array = np.frombuffer(data.take(len(scalar_names) * 4), dtype=np.float32)
        scalars = {n: v for n, v in zip(scalar_names, scalar_array)}

        self.file_pi = file_pi
        self.move_index = int(scalars.pop("pos_index"))

        move_count = int(scalars.pop("game_length"))

        self.simulation = Simulation(
            index=int(scalars.pop("game_id")),
            start_file_pi=self.file_pi - self.move_index,
            move_count=move_count,
            includes_final=includes_final,
        )
        self.final_position = final_position

        self.zero_visits = int(scalars.pop("zero_visits"))
        self.available_mv_count = int(scalars.pop("available_mv_count"))

        self.played_mv = map_none(scalars.pop("played_mv", None), int)
        self.is_full_search = map_none_or(scalars.pop("is_full_search", None), bool, True)
        self.is_final = map_none_or(scalars.pop("is_final_position", None), bool, False)
        self.is_terminal = map_none_or(scalars.pop("is_terminal", None), bool, False)
        self.hit_move_limit = map_none(scalars.pop("hit_move_limit", None), bool)
        self.is_post_final = False

        self.kdl_policy = float(scalars.pop("kdl_policy"))

        self.final_v = float(scalars.pop("final_v"))
        self.zero_v = float(scalars.pop("zero_v"))
        self.net_v = float(scalars.pop("net_v"))

        self.final_wdl = np.array([scalars.pop("final_wdl_w"), scalars.pop("final_wdl_d"), scalars.pop("final_wdl_l")])
        self.zero_wdl = np.array([scalars.pop("zero_wdl_w"), scalars.pop("zero_wdl_d"), scalars.pop("zero_wdl_l")])
        self.net_wdl = np.array([scalars.pop("net_wdl_w"), scalars.pop("net_wdl_d"), scalars.pop("net_wdl_l")])

        self.final_moves_left = float(scalars.pop("final_moves_left", move_count - self.move_index))
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


class PostFinalPosition:
    def __init__(self, final_position: Position):
        game = final_position.game
        self.game = game

        self.available_mv_count = 0

        self.input_scalars = np.full(game.input_scalar_channels, np.nan, np.float32)
        self.input_bools = np.full(game.input_bool_shape, np.nan, np.uint8)

        self.policy_indices = np.zeros(0, dtype=np.int32)
        self.policy_values = np.zeros(0, dtype=np.float32)

        self.move_index = -1
        self.file_pi = -1
        self.simulation = Simulation(
            index=-1,
            start_file_pi=-1,
            move_count=-1,
            includes_final=False,
        )
        self.final_position = final_position

        # pick a random move to teach that any more stays in the terminal state
        mv_size = prod(game.input_mv_shape)
        self.played_mv = random.randrange(mv_size)

        # TODO is this right? we "extremify" the values here
        #  doesn't really matter since usually we train on terminal values
        self.final_wdl = final_position.final_wdl
        self.zero_wdl = final_position.final_wdl
        self.net_wdl = final_position.final_wdl
        self.final_v = final_position.final_v
        self.zero_v = final_position.final_v
        self.net_v = final_position.final_v
        self.final_moves_left = 0.0
        self.zero_moves_left = 0.0
        self.net_moves_left = 0.0

        # TODO what about these values?
        self.is_full_search = False
        self.is_final = final_position.is_final
        self.is_terminal = final_position.is_terminal
        self.hit_move_limit = final_position.hit_move_limit
        self.is_post_final = True


class PositionBatch:
    def __init__(self, game: Game, positions: List[Position], include_final_for_each: bool, pin_memory: bool):
        self.max_available_moves = max(p.available_mv_count if p is not None else 0 for p in positions)

        input_full = torch.empty(len(positions), *game.full_input_shape, pin_memory=pin_memory)
        all_wdls = torch.empty(len(positions), 3 * 3, pin_memory=pin_memory)
        all_values = torch.empty(len(positions), 3, pin_memory=pin_memory)
        all_moves_left = torch.empty(len(positions), 3, pin_memory=pin_memory)

        final_input_full = torch.empty(len(positions), *game.full_input_shape, pin_memory=pin_memory) \
            if include_final_for_each else None

        # positions with less available moves get padded extra indices 0 and values -1
        policy_indices = torch.zeros(len(positions), self.max_available_moves, dtype=torch.int64, pin_memory=pin_memory)
        policy_values = torch.empty(len(positions), self.max_available_moves, pin_memory=pin_memory)
        policy_values.fill_(-1)

        played_mv = torch.empty(len(positions), dtype=torch.int64, pin_memory=pin_memory)
        sim_index = torch.empty(len(positions), dtype=torch.int64, pin_memory=pin_memory)
        move_index = torch.empty(len(positions), dtype=torch.int64, pin_memory=pin_memory)
        file_pi = torch.empty(len(positions), dtype=torch.int64, pin_memory=pin_memory)

        is_terminal = torch.empty(len(positions), dtype=torch.bool, pin_memory=pin_memory)
        is_final = torch.empty((len(positions)), dtype=torch.bool, pin_memory=pin_memory)
        is_post_final = torch.empty((len(positions)), dtype=torch.bool, pin_memory=pin_memory)

        if game.input_mv_channels is not None:
            played_mv_full = torch.zeros(len(positions), *game.input_mv_shape, pin_memory=pin_memory)
        else:
            played_mv_full = None

        for i, p in enumerate(positions):
            assert p.game == game

            write_input(game, input_full[i, :, :, :], p)
            if include_final_for_each:
                assert p.final_position is not None
                write_input(game, final_input_full[i, :, :, :], p.final_position)

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

            played_mv[i] = p.played_mv
            move_index[i] = p.move_index
            file_pi[i] = p.file_pi
            sim_index[i] = p.simulation.index

            if game.input_mv_channels is not None:
                played_mv_full[i, :, :, :] = torch.from_numpy(game.encode_mv(p.played_mv))

            is_terminal[i] = p.is_terminal
            is_final[i] = p.is_final
            is_post_final[i] = p.is_post_final

        self.input_full = input_full.to(DEVICE)
        self.final_input_full = final_input_full.to(DEVICE) if include_final_for_each else None

        self.policy_indices = policy_indices.to(DEVICE)
        self.policy_values = policy_values.to(DEVICE)

        self.played_mv = played_mv.to(DEVICE)
        self.move_index = move_index.to(DEVICE)
        self.file_pi = file_pi.to(DEVICE)
        self.sim_index = sim_index.to(DEVICE)
        self.played_mv_full = played_mv_full.to(DEVICE) if played_mv_full is not None else None

        self.is_terminal = is_terminal.to(DEVICE)
        self.is_final = is_final.to(DEVICE)
        self.is_post_final = is_post_final.to(DEVICE)

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


def write_input(game: Game, target, position: Position):
    target[:game.input_scalar_channels, :, :] = torch.tensor(position.input_scalars) \
        .view(-1, 1, 1).expand(*game.input_scalar_shape)
    target[game.input_scalar_channels:, :, :] = torch.tensor(position.input_bools) \
        .view(*game.input_bool_shape)


class UnrolledPositionBatch:
    def __init__(
            self,
            game: Game,
            unroll_steps: int,
            include_final_for_each: bool,
            batch_size: int,
            chains: List[List[Position]],
            pin_memory: bool
    ):
        assert unroll_steps >= 0, "Negative unroll steps don't make sense"
        for chain in chains:
            assert len(chain) == unroll_steps + 1, f"Expected {unroll_steps + 1} positions, got chain with {len(chain)}"

        positions_by_step = [[] for _ in range(unroll_steps + 1)]

        for chain in chains:
            last_position = None

            for si, p in enumerate(chain):
                if p is not None:
                    positions_by_step[si].append(p)
                    last_position = p
                else:
                    assert last_position is not None, "Each chain must contain at least one position"
                    positions_by_step[si].append(PostFinalPosition(last_position))

        self.unroll_steps = unroll_steps
        self.batch_size = batch_size
        self.positions = [
            PositionBatch(game, positions, include_final_for_each, pin_memory)
            for positions in positions_by_step
        ]

    def __len__(self):
        return self.batch_size
