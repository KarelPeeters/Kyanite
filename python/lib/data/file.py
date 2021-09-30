import json
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import List

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

        self.f = open(bin_path, "rb", buffering=0)

        self.position_offsets_offset = self.meta["position_offsets_offset"]
        self.f.seek(self.position_offsets_offset)
        self.offsets = np.frombuffer(self.f.read(8 * self.meta["position_count"]), dtype=np.int64)

    def __len__(self):
        return self.meta["position_count"]

    def __getitem__(self, item) -> Position:
        assert isinstance(item, int)

        start_offset = self.offsets[item]
        end_offset = self.offsets[item + 1] if item + 1 < len(self.offsets) else self.position_offsets_offset

        # TODO this combination of seek+read makes this non-thread-safe
        self.f.seek(start_offset)
        return Position(self.game, self.f.read(end_offset - start_offset))

    def close(self):
        self.f.close()


class DataFileLoader:
    def __init__(self, files: List[DataFile]):
        self.files = files
        self.pool = ThreadPool()
