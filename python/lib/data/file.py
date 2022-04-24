import json
import os
from pathlib import Path
from threading import Lock
from typing import overload, List, Union

import numpy as np

from lib.data.position import Position
from lib.games import Game

OFFSET_SIZE_IN_BYTES = 8


class DataFileInfo:
    def __init__(self, game: Game, meta: dict, bin_path: Path, off_path: Path, final_offset: int, timestamp: float):
        assert meta["game"] == game.name, f"Expected game {game.name}, got {meta['game']}"
        assert meta["input_bool_shape"] == list(game.input_bool_shape)
        assert meta["input_scalar_count"] == game.input_scalar_channels
        assert meta["policy_shape"] == list(game.policy_shape)

        self.game = game
        self.meta = meta
        self.bin_path = bin_path
        self.off_path = off_path
        self.final_offset = final_offset
        self.timestamp = timestamp

        self.position_count = meta["position_count"]
        self.includes_terminal_positions = meta.get("includes_terminal_positions", False)
        self.game_count = meta["game_count"]
        self.min_game_length = meta["min_game_length"]
        self.max_game_length = meta["max_game_length"]
        self.root_wdl = meta.get("root_wdl")

        self.mean_game_length = (self.position_count - self.includes_terminal_positions * self.game_count) / self.game_count

        self.scalar_names = meta["scalar_names"]


class DataFile:
    def __init__(self, info: DataFileInfo, bin_handle, off_handle):
        assert isinstance(info, DataFileInfo)

        self.info = info
        self.bin_handle = bin_handle
        self.off_handle = off_handle
        self.lock = Lock()

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
        offset_count = off_handle.tell() / OFFSET_SIZE_IN_BYTES

        bin_handle.seek(0, os.SEEK_END)
        final_offset = bin_handle.tell()

        # wrap everything up
        info = DataFileInfo(game, meta, bin_path, off_path, final_offset, timestamp)
        assert info.position_count == offset_count, f"Mismatch between offset and position counts for '{path}'"
        return DataFile(info, bin_handle, off_handle)

    def with_new_handle(self) -> 'DataFile':
        return DataFile(
            self.info,
            random_access_handle(self.info.bin_path),
            random_access_handle(self.info.off_path),
        )

    def __len__(self):
        return self.info.position_count

    def __getitem__(self, item: Union[int, slice]) -> Union[Position, List[Position]]:
        if isinstance(item, slice):
            return [self[i] for i in range(len(self))[item]]

        assert isinstance(item, (int, np.intc)), f"Expected int, got {type(item)}"
        if not (0 <= item < len(self)):
            raise IndexError(f"Index {item} out of bounds in file with {len(self)} positions")

        # lock to ensure no other thread starts seeking to another position
        with self.lock:
            self.off_handle.seek(item * OFFSET_SIZE_IN_BYTES)

            if item == len(self) - 1:
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

    def close(self):
        self.bin_handle.close()
        self.off_handle.close()


def random_access_handle(path: Path):
    return open(path, "rb", buffering=0)
