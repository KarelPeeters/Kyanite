import bisect
from typing import List


class Schedule:
    def __init__(self, warmup: int, lrs: List[float], thresholds: List[int]):
        assert len(thresholds) + 1 == len(lrs)
        assert len(lrs) > 0
        assert sorted(thresholds) == thresholds

        self.warmup = warmup
        self.thresholds = thresholds
        self.rates = lrs

    def get(self, bi: int):
        if bi < self.warmup:
            return self.rates[0] * (bi + 1) / self.warmup

        i = bisect.bisect_right(self.thresholds, bi)
        return self.rates[i]

