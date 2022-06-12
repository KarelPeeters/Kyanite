import json
import os
from functools import cached_property
from pathlib import Path
from threading import RLock
from typing import BinaryIO, Sequence, overload, Union

import numpy as np

from lib.data.position import Position, Simulation
from lib.games import Game

OFFSET_SIZE_IN_BYTES = 8


class DataFileInfo:
    def __init__(self, game: Game, meta: dict, bin_path: Path, off_path: Path, final_offset: int, timestamp: float):
        assert meta.pop("game") == game.name, f"Expected game {game.name}, got {meta['game']}"
        assert meta.pop("input_bool_shape") == list(game.input_bool_shape)
        assert meta.pop("input_scalar_count") == game.input_scalar_channels
        assert meta.pop("policy_shape") == list(game.policy_shape)

        self.game = game
        self.bin_path = bin_path
        self.off_path = off_path
        self.final_offset = final_offset
        self.timestamp = timestamp

        self.position_count = meta.pop("position_count")
        self.includes_final_positions = meta.pop("includes_terminal_positions", False)
        self.simulation_count = meta.pop("game_count")
        self.min_simulation_length = meta.pop("min_game_length")
        self.max_simulation_length = meta.pop("max_game_length")
        self.root_wdl = meta.pop("root_wdl", None)
        self.hit_move_limit = meta.pop("hit_move_limit", None)
        self.includes_simulation_start_indices = meta.pop("includes_game_start_indices", False)

        total_move_count = self.position_count - self.includes_final_positions * self.simulation_count
        self.mean_simulation_length = total_move_count / self.simulation_count

        self.scalar_names = meta.pop("scalar_names")

        assert len(meta) == 0, f"Leftover meta values: {meta}"


class DataFile:
    def __init__(self, info: DataFileInfo, bin_handle: BinaryIO, off_handle: BinaryIO):
        assert isinstance(info, DataFileInfo)

        self.info = info
        self.simulations = FileSimulationsView(self, range(self.info.simulation_count))
        self.positions = FilePositionsView(self, range(self.info.position_count))

        # this lock is specifically for seeking in both handles
        self.lock = RLock()
        self.bin_handle = bin_handle
        self.off_handle = off_handle

    @staticmethod
    def open(game: Game, path: str) -> 'DataFile':
        path = Path(path)
        json_path = path.with_suffix(".json").absolute()
        bin_path = path.with_suffix(".bin").absolute()
        off_path = path.with_suffix(".off").absolute()

        for p in [json_path, off_path, bin_path]:
            if not p.exists():
                raise FileNotFoundError(f"{p} does not exist")

        with open(json_path, "r") as json_f:
            meta = json.loads(json_f.read())
        timestamp = os.path.getmtime(json_path)

        bin_handle = random_access_handle(bin_path)

        # for large datasets even the offsets don't fit into RAM, so we're reading them from disk as we need them
        off_handle = random_access_handle(off_path)
        off_handle.seek(0, os.SEEK_END)
        off_len_bytes = off_handle.tell()

        bin_handle.seek(0, os.SEEK_END)
        final_offset = bin_handle.tell()

        info = DataFileInfo(game, meta, bin_path, off_path, final_offset, timestamp)

        if info.includes_simulation_start_indices:
            expected_off_len_bytes = OFFSET_SIZE_IN_BYTES * (info.position_count + info.simulation_count)
        else:
            expected_off_len_bytes = OFFSET_SIZE_IN_BYTES * info.position_count
        assert expected_off_len_bytes == off_len_bytes, f"Mismatch in offset size, expected {expected_off_len_bytes} but got {off_len_bytes}"

        return DataFile(info, bin_handle, off_handle)

    def with_new_handles(self) -> 'DataFile':
        # TODO do we actually need any of this?
        #   typically we're sampling from many files at once so there shouldn't be too much locking
        return DataFile(
            self.info,
            random_access_handle(self.info.bin_path),
            random_access_handle(self.info.off_path),
        )

    def load_position(self, pi: int) -> Position:
        with self.lock:
            self.off_handle.seek(pi * OFFSET_SIZE_IN_BYTES)

            if pi == self.info.position_count - 1:
                off_bytes = self.off_handle.read(OFFSET_SIZE_IN_BYTES)
                start_offset = int.from_bytes(off_bytes, "little")
                end_offset = self.info.final_offset
            else:
                off_bytes = self.off_handle.read(2 * OFFSET_SIZE_IN_BYTES)
                start_offset = int.from_bytes(off_bytes[:OFFSET_SIZE_IN_BYTES], "little")
                end_offset = int.from_bytes(off_bytes[OFFSET_SIZE_IN_BYTES:], "little")

            self.bin_handle.seek(start_offset)
            data = self.bin_handle.read(end_offset - start_offset)

        return Position(self.info.game, pi, self.info.includes_final_positions, self.info.scalar_names, data)

    def load_simulation(self, si: int) -> Simulation:
        is_final_simulation = si == self.info.simulation_count - 1

        with self.lock:
            if self.info.includes_simulation_start_indices:
                self.off_handle.seek((self.info.position_count + si) * OFFSET_SIZE_IN_BYTES)

                if is_final_simulation:
                    index_bytes = self.off_handle.read(OFFSET_SIZE_IN_BYTES)
                    start_pi = int.from_bytes(index_bytes, "little")
                    end_pi = self.info.position_count - 1
                else:
                    index_bytes = self.off_handle.read(OFFSET_SIZE_IN_BYTES * 2)
                    start_pi = int.from_bytes(index_bytes[:OFFSET_SIZE_IN_BYTES], "little")
                    end_pi = int.from_bytes(index_bytes[OFFSET_SIZE_IN_BYTES:], "little") - 1
            else:
                start_indices = self._simulation_start_indices

                start_pi = start_indices[si]

                if is_final_simulation:
                    end_pi = self.info.position_count - 1
                else:
                    end_pi = start_indices[si + 1] - 1

        return Simulation(
            index=si,
            start_file_pi=start_pi,
            move_count=(end_pi - start_pi + 1 - self.info.includes_final_positions),
            includes_terminal=self.info.includes_final_positions,
        )

    @cached_property
    def _simulation_start_indices(self):
        """
        The start position index for each simulation, calculated based on the positions themselves.
        Useful for backwards compatibility for files without includes_game_start_indices.
        """

        starts = np.empty(self.info.simulation_count, dtype=int)
        pi = 0

        for gi in range(self.info.simulation_count):
            starts[gi] = pi

            position = self.load_position(pi)
            assert position.move_index == 0
            pi += position.simulation.position_count

        return starts

    def close(self):
        self.bin_handle.close()
        self.off_handle.close()


class FileSimulationsView(Sequence[Simulation]):
    def __init__(self, file: DataFile, si_range: range):
        self.file = file
        self.si_range = si_range

    def __len__(self):
        return len(self.si_range)

    @overload
    def __getitem__(self, si: int) -> Simulation:
        pass

    @overload
    def __getitem__(self, si_slice: slice) -> 'FileSimulationsView':
        pass

    @cached_property
    def positions(self) -> 'FilePositionsView':
        assert self.si_range.step == 1, "Cannot get positions for simulation slice with step, since it would be non-affine"

        if len(self) == 0:
            return FilePositionsView(self.file, range(self.si_range.start, self.si_range.start))

        start_pi = self[0].start_file_pi
        end_pi = self[-1].end_file_pi
        return FilePositionsView(self.file, range(start_pi, end_pi))

    def __getitem__(self, item: Union[int, slice]) -> Union[Simulation, 'FileSimulationsView']:
        if isinstance(item, slice):
            return FileSimulationsView(self.file, self.si_range[item])

        if not (-len(self) <= item < len(self)):
            raise IndexError(f"Index {item} out of bounds for {len(self)} simulations")
        return self.file.load_simulation(self.si_range[item])

    def with_new_handles(self) -> 'FileSimulationsView':
        return FileSimulationsView(self.file.with_new_handles(), self.si_range)


class FilePositionsView(Sequence[Position]):
    def __init__(self, file: DataFile, pi_range: range):
        self.file = file
        self.pi_range = pi_range

    def __len__(self):
        return len(self.pi_range)

    @overload
    def __getitem__(self, pi: int) -> Position:
        pass

    @overload
    def __getitem__(self, pi_slice: slice) -> 'FilePositionsView':
        pass

    def __getitem__(self, item: Union[int, slice]) -> Union[Position, 'FilePositionsView']:
        if isinstance(item, slice):
            return FilePositionsView(self.file, self.pi_range[item])

        if not (-len(self) <= item < len(self)):
            raise IndexError(f"Index {item} out of bounds for {len(self)} positions")
        return self.file.load_position(self.pi_range[item])

    def with_new_handles(self) -> 'FilePositionsView':
        return FilePositionsView(self.file.with_new_handles(), self.pi_range)


def random_access_handle(path: Path) -> BinaryIO:
    return open(path, "rb", buffering=0)
