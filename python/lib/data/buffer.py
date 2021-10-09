from multiprocessing.pool import ThreadPool
from typing import List

import numpy as np

from lib.data.file import DataFile
from lib.data.position import PositionBatch, Position
from lib.games import Game
from lib.util import PIN_MEMORY


class FileList:
    def __init__(self, game: Game, files: List[DataFile], pool: ThreadPool):
        self.game = game
        self.pool = pool

        self.files = files
        self.file_ends = np.cumsum(np.array([len(f) for f in self.files]))

    def __len__(self):
        return self.file_ends[-1] if len(self.file_ends) else 0

    def __getitem__(self, indices: np.array) -> PositionBatch:
        def load_single(i) -> Position:
            assert 0 <= i < len(self)

            fi = np.searchsorted(self.file_ends, i, "right")
            if fi == 0:
                pi = i
            else:
                pi = i - self.file_ends[fi - 1]
            return self.files[fi][pi]

        positions = self.pool.map(load_single, indices)
        return PositionBatch(self.game, positions, pin_memory=PIN_MEMORY)

    # TODO caching? maybe return a new class here?
    def sample_batch(self, size: int) -> PositionBatch:
        return self[np.random.randint(len(self), size=size)]
