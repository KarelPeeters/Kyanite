import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

import numpy as np

from lib.growable_array import GrowableArray

Key = Tuple[str, str]


@dataclass
class LoggerData:
    axis: np.array
    values: Dict[Key, np.array]


class Logger:
    def __init__(self):
        self.curr_batch = -1
        self.data: Dict[Key, GrowableArray] = {}

    def start_batch(self):
        self.curr_batch += 1
        for a in self.data.values():
            a.append(np.NaN)

    def log(self, group: str, name: str, value):
        assert self.curr_batch >= 0, "No batch has been started yet"
        key = (group, name)

        if key in self.data:
            a = self.data[key]
        else:
            a = GrowableArray(initial_values=np.full(self.curr_batch + 1, np.NaN))
            self.data[key] = a

        assert np.isnan(a[self.curr_batch]), f"Key {key} was already logged during this batch"
        a[self.curr_batch] = value

    def finished_data(self) -> LoggerData:
        return LoggerData(
            axis=np.arange(self.curr_batch),
            values={k: v.values[:self.curr_batch] for k, v in self.data.items()}
        )

    def save(self, path: str):
        path = Path(path)
        assert path.suffix == ".npz", f"Log save path should have extension .npz, got {path}"

        data = {
            "curr_batch": self.curr_batch,
            "keys": list(self.data.keys()),
            "values": [v.values for v in self.data.values()],
        }

        tmp_path = path.with_suffix(".tmp.npz")
        np.savez(tmp_path, **data)
        os.replace(tmp_path, path)

    @staticmethod
    def load(path) -> 'Logger':
        data = np.load(path)

        result = Logger()
        result.curr_batch = data["curr_batch"]
        result.data = {
            tuple(k): GrowableArray(v)
            for k, v in zip(data["keys"], data["values"])
        }

        return result
