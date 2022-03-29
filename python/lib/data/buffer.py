import random
from threading import Thread
from typing import List, Optional, Union

import numpy as np

from lib.data.file import DataFile
from lib.data.position import PositionBatch, UnrolledPositionBatch
from lib.games import Game
from lib.queue import CQueue, CQueueClosed
from lib.util import PIN_MEMORY


class FileListSampler:
    def __init__(
            self, game: Game, files: List[DataFile],
            batch_size: int, unroll_steps: Optional[int],
            threads: int = 2
    ):
        self.game = game
        self.unroll_steps = unroll_steps

        self.files = files
        self.file_ends = np.cumsum(np.array([len(f) for f in self.files]))

        assert len(self.file_ends), "There must be at least one file"
        assert self.file_ends[-1] != 0, "All files are empty"

        self.batch_size = batch_size
        self.queue = CQueue(threads + 1)

        self.threads = [
            Thread(target=thread_main, args=(self, unroll_steps), daemon=True)
            for _ in range(threads)
        ]
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

    def next_batch_either(self) -> Union[PositionBatch, UnrolledPositionBatch]:
        if self.unroll_steps is None:
            return self.next_batch()
        else:
            return self.next_unrolled_batch()

    def next_batch(self) -> PositionBatch:
        assert self.unroll_steps is None, "This sampler does not sample simple batches"
        return self.queue.pop_blocking()

    def next_unrolled_batch(self) -> UnrolledPositionBatch:
        assert self.unroll_steps is not None, "This sampler does not sample unrolled batches"
        return self.queue.pop_blocking()


def thread_main(sampler: FileListSampler, unroll_steps: Optional[int]):
    files = [f.with_new_handle() for f in sampler.files]

    try:
        while True:
            if unroll_steps is None:
                sampler.queue.push_blocking(collect_simple_batch(sampler, files))
            else:
                sampler.queue.push_blocking(collect_unrolled_batch(sampler, files, unroll_steps))

    except CQueueClosed:
        for f in files:
            f.close()


def collect_simple_batch(sampler: FileListSampler, files: List[DataFile]):
    positions = []

    for _ in range(sampler.batch_size):
        i = random.randrange(len(sampler))
        fi, pi = sampler.split_index(i)
        positions.append(files[fi][pi])

    return PositionBatch(sampler.game, positions, PIN_MEMORY)


def collect_unrolled_batch(sampler: FileListSampler, files: List[DataFile], unroll_steps: int):
    chains = []

    for _ in range(sampler.batch_size):
        i = random.randrange(len(sampler))
        fi, pi = sampler.split_index(i)
        file = files[fi]

        first_position = file[pi]
        chain = [first_position]

        for ri in range(unroll_steps):
            ni = pi + 1 + ri
            if ni < len(file):
                next_position = file[ni]
                if next_position.game_id == first_position.game_id:
                    chain.append(next_position)
                    continue
            chain.append(None)

        chains.append(chain)

    return UnrolledPositionBatch(sampler.game, unroll_steps, chains, PIN_MEMORY)
