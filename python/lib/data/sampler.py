import random
from threading import Thread
from typing import Optional, Union

from lib.data.group import DataGroup
from lib.data.position import PositionBatch, UnrolledPositionBatch, Position
from lib.queue import CQueue, CQueueClosed
from lib.util import PIN_MEMORY


class PositionSampler:
    def __init__(
            self,
            group: DataGroup,
            batch_size: int,
            unroll_steps: Optional[int],
            include_final: bool,
            threads: int,
    ):
        self.group = group

        self.batch_size = batch_size
        self.unroll_steps = unroll_steps
        self.include_final = include_final

        self.queue = CQueue(threads + 1)

        self.threads = [
            Thread(target=thread_main, args=(self,), daemon=True)
            for _ in range(threads)
        ]
        for thread in self.threads:
            thread.start()

    def close(self):
        self.queue.close()

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


def thread_main(sampler: PositionSampler):
    group = sampler.group.with_new_handles()
    unroll_steps = sampler.unroll_steps

    try:
        while True:
            if unroll_steps is None:
                sampler.queue.push_blocking(collect_simple_batch(sampler, group))
            else:
                sampler.queue.push_blocking(collect_unrolled_batch(sampler, group, unroll_steps))

    except CQueueClosed:
        group.close()


def collect_simple_batch(sampler: PositionSampler, group: DataGroup):
    positions = []

    for _ in range(sampler.batch_size):
        _, p = sample_position(group, sampler.include_final)
        positions.append(p)

    return PositionBatch(group.game, positions, PIN_MEMORY)


def collect_unrolled_batch(sampler: PositionSampler, group: DataGroup, unroll_steps: int):
    chains = []

    for _ in range(sampler.batch_size):
        (first_pi, first_position) = sample_position(group, sampler.include_final)
        chain = [first_position]

        for ri in range(unroll_steps):
            pi = first_pi + 1 + ri
            expected_file_pi = first_position.file_pi + 1 + ri

            # we've reached the end of the group
            if pi >= len(group.positions):
                break
            position = group.positions[pi]

            # we've reached the end of the file
            if position.file_pi != expected_file_pi:
                break
            # we've reached the end of the simulation
            if position.simulation.index != first_position.simulation.index:
                break
            # maybe we don't want the final position
            if position.is_final_position and not sampler.include_final:
                break

            # finally we can include the position
            chain.append(position)

        # pad the chain until we've reached the desired unroll steps
        while len(chain) < unroll_steps + 1:
            chain.append(None)

        chains.append(chain)

    return UnrolledPositionBatch(group.game, unroll_steps, sampler.batch_size, chains, PIN_MEMORY)


def sample_position(group: DataGroup, include_final: bool) -> (int, Position):
    while True:
        pi = random.randrange(len(group.positions))
        pos = group.positions[pi]

        if pos.is_final_position and not include_final:
            continue

        return pi, pos
