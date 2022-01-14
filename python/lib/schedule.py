import bisect
from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Schedule(ABC):
    @abstractmethod
    def __call__(self, bi: int) -> float:
        pass


class WarmupSchedule(Schedule):
    def __init__(self, steps: int, inner: Schedule):
        assert steps >= 0

        self.steps = steps
        self.inner = inner

    def __call__(self, bi: int) -> float:
        # linearly go from 0 to the initial learning rate
        if bi < self.steps:
            return (bi / self.steps) * self.inner(0)

        return self.inner(bi - self.steps)


class FixedSchedule(Schedule):
    def __init__(self, values: List[float], periods: List[int]):
        assert len(values) == len(periods) + 1
        assert len(values) > 0

        self.thresholds = np.cumsum(periods)
        self.values = values

    def __call__(self, bi: int):
        i = bisect.bisect_right(self.thresholds, bi)
        return self.values[i]


class LinearSchedule(Schedule):
    def __init__(self, initial: float, final: float, steps: int):
        self.initial = initial
        self.final = final
        self.steps = steps

    def __call__(self, bi: int) -> float:
        return lerp_clipped(self.initial, self.final, bi / self.steps)


class ExpSchedule(Schedule):
    def __init__(self, initial: float, final: float, steps: int):
        self.initial = initial
        self.final = final
        self.steps = steps

    def __call__(self, bi: int) -> float:
        t = np.clip(bi / self.steps, 0, 1)
        return self.initial * (self.final / self.initial) ** t


class TriSchedule(Schedule):
    def __init__(self, min: float, max: float, cycle_steps: int):
        self.min = min
        self.max = max
        self.cycle_steps = cycle_steps

    def __call__(self, bi: int) -> float:
        t = (bi / self.cycle_steps) % 1.0
        p = min(2 * t, 2 - 2 * t)
        return lerp_clipped(self.min, self.max, p)


def lerp_clipped(a, b, t):
    t = np.clip(t, 0, 1)
    return (1 - t) * a + t * b
