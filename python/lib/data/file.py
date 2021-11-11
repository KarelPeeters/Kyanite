import json
import os
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np

from lib.data.position import Position
from lib.games import Game


class DataFileInfo:
    def __init__(self, game: Game, meta: dict, bin_path: Path, offsets: np.array, final_offset: int):
        assert meta["game"] == game.name
        assert meta["board_bool_planes"] == game.input_bool_channels
        assert meta["board_scalar_count"] == game.input_scalar_channels
        assert meta["policy_planes"] == game.policy_channels

        self.game = game
        self.meta = meta
        self.bin_path = bin_path
        self.offsets = offsets
        self.final_offset = final_offset

        self.loaded_position_count = len(offsets)

        self.position_count = meta["position_count"]
        self.game_count = meta["game_count"]
        self.min_game_length = meta["min_game_length"]
        self.max_game_length = meta["max_game_length"]
        self.root_wdl = meta.get("root_wdl")


class DataFile:
    def __init__(self, info: DataFileInfo, handle):
        assert isinstance(info, DataFileInfo)

        self.info = info
        self.handle = handle
        self.lock = Lock()

    @staticmethod
    def open(game: Game, path: str, max_positions: Optional[int]) -> 'DataFile':
        path = Path(path)
        json_path = path.with_suffix(".json")
        offset_path = path.with_suffix(".off")
        bin_path = path.with_suffix(".bin")

        for p in [json_path, offset_path, bin_path]:
            assert p.exists(), f"{p} does not exist"

        with open(json_path, "r") as json_f:
            meta = json.loads(json_f.read())

        loaded_position_count = meta["position_count"]
        if max_positions is not None:
            loaded_position_count = min(loaded_position_count, max_positions)

        with open(offset_path, "rb") as off_f:
            offset_byte_count = loaded_position_count * 8
            offset_bytes = off_f.read(offset_byte_count)

        assert len(offset_bytes) == offset_byte_count, f"Offset file {offset_path} too short"
        offsets = np.frombuffer(offset_bytes, dtype=np.int64)

        handle = random_access_handle(bin_path)

        handle.seek(0, os.SEEK_END)
        final_offset = handle.tell()

        # wrap everything up
        info = DataFileInfo(game, meta, bin_path, offsets, final_offset)
        return DataFile(info, handle)

    def with_new_handle(self) -> 'DataFile':
        return DataFile(self.info, random_access_handle(self.info.bin_path))

    def __len__(self):
        return self.info.loaded_position_count

    def __getitem__(self, item: int) -> Position:
        assert item < len(self), f"Index {item} out of bounds in file with {len(self)} loaded positions"

        start_offset = self.info.offsets[item]
        end_offset = self.info.offsets[item + 1] if item < len(self) - 1 else self.info.final_offset

        # lock to ensure no other thread starts seeking to another position
        with self.lock:
            self.handle.seek(start_offset)
            data = self.handle.read(end_offset - start_offset)

        return Position(self.info.game, data)

    def close(self):
        self.handle.close()


def random_access_handle(path: Path):
    return open(path, "rb", buffering=0)
