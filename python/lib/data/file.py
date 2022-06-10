import json
import os
from pathlib import Path
from threading import RLock
from typing import List, Union, BinaryIO

import numpy as np

from lib.data.position import Position, GameSimulation
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
        self.includes_terminal_positions = meta.pop("includes_terminal_positions", False)
        self.game_count = meta.pop("game_count")
        self.min_game_length = meta.pop("min_game_length")
        self.max_game_length = meta.pop("max_game_length")
        self.root_wdl = meta.pop("root_wdl", None)
        self.hit_move_limit = meta.pop("hit_move_limit", None)
        self.includes_game_start_indices = meta.pop("includes_game_start_indices", False)

        total_move_count = self.position_count - self.includes_terminal_positions * self.game_count
        self.mean_game_length = total_move_count / self.game_count

        self.scalar_names = meta.pop("scalar_names")

        assert len(meta) == 0, f"Leftover meta values: {meta}"


class DataFile:
    def __init__(self, info: DataFileInfo, bin_handle: BinaryIO, off_handle: BinaryIO):
        assert isinstance(info, DataFileInfo)

        self.info = info
        self.games = FileGames(self)
        self.positions = FilePositions(self)

        # this lock is specifically for seeking in both handles
        self.lock = RLock()
        self.bin_handle = bin_handle
        self.off_handle = off_handle

        self._cached_game_indices = None

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

        if info.includes_game_start_indices:
            expected_off_len_bytes = OFFSET_SIZE_IN_BYTES * (info.position_count + info.game_count)
        else:
            expected_off_len_bytes = OFFSET_SIZE_IN_BYTES * info.position_count
        assert expected_off_len_bytes == off_len_bytes, f"Mismatch in offset size, expected {expected_off_len_bytes} but got {off_len_bytes}"

        return DataFile(info, bin_handle, off_handle)

    def with_new_handle(self) -> 'DataFile':
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

        return Position(self.info.game, self.info.scalar_names, data)

    def load_simulation(self, gi: int) -> GameSimulation:
        is_final_game = gi == self.info.game_count - 1

        with self.lock:
            if self.info.includes_game_start_indices:
                self.off_handle.seek((self.info.position_count + gi) * OFFSET_SIZE_IN_BYTES)

                if is_final_game:
                    index_bytes = self.off_handle.read(OFFSET_SIZE_IN_BYTES)
                    start_index = int.from_bytes(index_bytes, "little")
                    end_index = self.info.position_count - 1
                else:
                    index_bytes = self.off_handle.read(OFFSET_SIZE_IN_BYTES * 2)
                    start_index = int.from_bytes(index_bytes[:OFFSET_SIZE_IN_BYTES], "little")
                    end_index = int.from_bytes(index_bytes[OFFSET_SIZE_IN_BYTES:], "little") - 1
            else:
                game_starts = self._game_start_indices()

                start_index = game_starts[gi]

                if is_final_game:
                    end_index = self.info.position_count - 1
                else:
                    end_index = game_starts[gi + 1] - 1

        return GameSimulation(
            start_pi=start_index,
            end_pi=end_index,
            length=end_index - start_index,
            id=gi,
        )

    def _game_start_indices(self):
        if self._cached_game_indices is not None:
            return self._cached_game_indices

        starts = np.empty(self.info.game_count, dtype=int)
        pi = 0

        for gi in range(self.info.game_count):
            starts[gi] = pi

            position = self.load_position(pi)
            assert position.pos_index == 0
            pi += position.game_length + self.info.includes_terminal_positions

        self._cached_game_indices = starts
        return starts

    def close(self):
        self.bin_handle.close()
        self.off_handle.close()


class FilePositions:
    def __init__(self, file: DataFile):
        self.file = file

    def __len__(self):
        return self.file.info.position_count

    def __getitem__(self, item: Union[int, slice]) -> Union[Position, List[Position]]:
        if isinstance(item, slice):
            return [self[i] for i in range(len(self))[item]]
        if not (0 <= item < len(self)):
            raise IndexError(f"Index {item} out of bounds for {len(self)} positions")

        return self.file.load_position(item)


class FileGames:
    def __init__(self, file: DataFile):
        self.file = file

    def __len__(self):
        return self.file.info.game_count

    def __getitem__(self, item: Union[int, slice]) -> Union[GameSimulation, List[GameSimulation]]:
        if isinstance(item, slice):
            return [self[i] for i in range(len(self))[item]]
        if not (0 <= item < len(self)):
            raise IndexError(f"Index {item} out of bounds for {len(self)} games")

        return self.file.load_simulation(item)


def random_access_handle(path: Path) -> BinaryIO:
    return open(path, "rb", buffering=0)
