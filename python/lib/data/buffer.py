import random
from threading import Thread
from typing import List

import numpy as np

from lib.data.file import DataFile
from lib.data.position import PositionBatch
from lib.games import Game
from lib.queue import CQueue, CQueueClosed
from lib.util import PIN_MEMORY


class FileListSampler:
    def __init__(self, game: Game, files: List[DataFile], batch_size: int, threads: int = 2):
        self.game = game

        self.files = files
        self.file_ends = np.cumsum(np.array([len(f) for f in self.files]))

        assert len(self.file_ends), "There must be at least one file"
        assert self.file_ends[-1] != 0, "All files are empty"

        self.batch_size = batch_size
        self.queue = CQueue(threads + 1)

        self.threads = [Thread(target=thread_main, args=(self,)) for _ in range(threads)]
        for thread in self.threads:
            thread.start()

    def total_position_count(self):
        return self.file_ends[-1] if len(self.file_ends) else 0

    def split_index(self, i: int) -> (int, int):
        assert 0 <= i < len(self)
        fi = np.searchsorted(self.file_ends, i, "right")
        if fi == 0:
            pi = i
        else:
            pi = i - self.file_ends[fi - 1]
        return fi, pi

    def close(self):
        self.queue.close()

    def __len__(self):
        return self.file_ends[-1] if len(self.file_ends) else 0

    def __getitem__(self, i: int):
        (fi, pi) = self.split_index(i)
        return self.files[fi][pi]

    def next_batch(self):
        return self.queue.pop_blocking()


def thread_main(sampler: FileListSampler):
    files = [f.with_new_handle() for f in sampler.files]

    try:
        while True:
            positions = []

            for _ in range(sampler.batch_size):
                i = random.randrange(len(sampler))
                fi, pi = sampler.split_index(i)

                positions.append(files[fi][pi])

            batch = PositionBatch(sampler.game, positions, PIN_MEMORY)
            sampler.queue.push_blocking(batch)
    except CQueueClosed:
        for f in files:
            f.close()
