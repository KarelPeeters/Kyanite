import json
from pathlib import Path
from threading import Lock

import numpy as np

from lib.data.position import Position
from lib.games import Game


class DataFileInfo:
    def __init__(self, game: Game, meta: dict, bin_path: Path, offsets: np.array, offsets_offset: int):
        assert meta["game"] == game.name
        assert meta["board_bool_planes"] == game.input_bool_channels
        assert meta["board_scalar_count"] == game.input_scalar_channels
        assert meta["policy_planes"] == game.policy_channels

        self.game = game
        self.meta = meta
        self.bin_path = bin_path
        self.offsets = offsets
        self.offsets_offset = offsets_offset

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
    def open(game: Game, path: str) -> 'DataFile':
        path = Path(path)
        json_path = path.with_suffix(".json")
        bin_path = path.with_suffix(".bin")

        assert json_path.exists(), f"{json_path} does not exist"
        assert bin_path.exists(), f"{bin_path} does not exist"

        with open(json_path, "r") as json_f:
            meta = json.loads(json_f.read())

        handle = random_access_handle(bin_path)

        # read the offsets
        position_offsets_offset = meta["position_offsets_offset"]
        handle.seek(position_offsets_offset)

        offsets_byte_count = 8 * meta["position_count"]
        offsets_bytes = handle.read(offsets_byte_count)
        assert len(offsets_bytes) == offsets_byte_count, f"File {path} too short"
        offsets = np.frombuffer(offsets_bytes, dtype=np.int64)

        # wrap everything up
        info = DataFileInfo(game, meta, bin_path, offsets, position_offsets_offset)
        return DataFile(info, handle)

    def with_new_handle(self) -> 'DataFile':
        return DataFile(self.info, random_access_handle(self.info.bin_path))

    def __len__(self):
        return self.info.position_count

    def __getitem__(self, item: int) -> Position:
        offsets = self.info.offsets

        start_offset = offsets[item]
        end_offset = offsets[item + 1] if item + 1 < len(offsets) else self.info.offsets_offset

        with self.lock:
            self.handle.seek(start_offset)
            data = self.handle.read(end_offset - start_offset)

        return Position(self.info.game, data)

    def close(self):
        self.handle.close()


def random_access_handle(path: Path):
    return open(path, "rb", buffering=0)
