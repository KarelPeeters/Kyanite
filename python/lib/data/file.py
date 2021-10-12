import json
from pathlib import Path
from threading import Lock

import numpy as np

from lib.data.position import Position
from lib.games import Game


class DataFile:
    def __init__(self, game: Game, path: str):
        self.game = game

        path = Path(path)
        json_path = path.with_suffix(".json")
        bin_path = path.with_suffix(".bin")

        assert json_path.exists(), f"{json_path} does not exist"
        assert bin_path.exists(), f"{bin_path} does not exist"

        with open(json_path, "r") as json_f:
            self.meta = json.loads(json_f.read())

        assert self.meta["game"] == game.name
        assert self.meta["board_bool_planes"] == game.input_bool_channels
        assert self.meta["board_scalar_count"] == game.input_scalar_channels
        assert self.meta["policy_planes"] == game.policy_channels

        self.position_count = self.meta["position_count"]
        self.game_count = self.meta["game_count"]
        self.min_game_length = self.meta["min_game_length"]
        self.max_game_length = self.meta["max_game_length"]

        self.f = open(bin_path, "rb", buffering=0)

        self.lock = Lock()

        with self.lock:
            self.position_offsets_offset = self.meta["position_offsets_offset"]
            self.f.seek(self.position_offsets_offset)

            offsets_byte_count = 8 * self.meta["position_count"]
            offsets_bytes = self.f.read(offsets_byte_count)
            assert len(offsets_bytes) == offsets_byte_count, f"File {path} too short"

        self.offsets = np.frombuffer(offsets_bytes, dtype=np.int64)

    def __len__(self):
        return self.meta["position_count"]

    def __getitem__(self, item: int) -> Position:
        start_offset = self.offsets[item]
        end_offset = self.offsets[item + 1] if item + 1 < len(self.offsets) else self.position_offsets_offset

        with self.lock:
            self.f.seek(start_offset)
            return Position(self.game, self.f.read(end_offset - start_offset))

    def close(self):
        self.f.close()
