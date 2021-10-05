import bisect
from abc import ABC, abstractmethod
from typing import List


class Schedule(ABC):
    @abstractmethod
    def get(self, bi: int) -> float:
        pass


class FixedSchedule(Schedule):
    def __init__(self, warmup: int, values: List[float], thresholds: List[int]):
        assert len(thresholds) + 1 == len(values)
        assert len(values) > 0
        assert sorted(thresholds) == thresholds

        self.warmup = warmup
        self.thresholds = thresholds
        self.values = values

    def get(self, bi: int):
        if bi < self.warmup:
            return self.values[0] * (bi + 1) / self.warmup

        i = bisect.bisect_right(self.thresholds, bi)
        return self.values[i]


class LinearSchedule(Schedule):
    def __init__(self, initial: float, final: float, steps: int):
        self.initial = initial
        self.final = final
        self.steps = steps

    def get(self, bi: int) -> float:
        t = bi / self.steps
        if t <= 1:
            return (1 - t) * self.initial + t * self.final

        return self.final
