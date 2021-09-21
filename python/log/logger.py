from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class FinishedLogData:
    gen_keys: List[Tuple[str, str]]
    batch_keys: List[Tuple[str, str]]

    gen_data: np.array
    gen_average_data: np.array

    batch_axis: np.array
    batch_data: np.array


# TODO serialize and deserialize log data
class Logger:
    def __init__(self):
        self.gen_keys = None
        self.batch_keys = None

        self.gen_data: Optional[GrowableArray] = None
        self.gen_average_data: Optional[GrowableArray] = None
        self.batch_data: Optional[GrowableArray] = None
        self.batch_axis = GrowableArray(1)

        self._curr_gen_data = None
        self._curr_gen_batch_data = None
        self._curr_batch_data = None

    def get_finished_data(self) -> FinishedLogData:
        # we don't copy anything here since every we pass is immutable
        return FinishedLogData(
            gen_keys=self.gen_keys, batch_keys=self.batch_keys,
            gen_data=self.gen_data.values, gen_average_data=self.gen_average_data.values,
            batch_axis=self.batch_axis.values.squeeze(1), batch_data=self.batch_data.values,
        )

    def start_gen(self):
        assert self._curr_gen_data is None, "The previous gen is still in progress"
        self._curr_gen_data = {}

    def start_batch(self):
        assert self._curr_gen_data is not None, "No generation in progress"
        assert self._curr_batch_data is None, "The previous batch is still in progress"
        self._curr_batch_data = {}

    def finish_gen(self):
        assert self._curr_gen_data is not None, "No generation was started"
        assert self._curr_batch_data is None, "A batch is still in progress"

        # store log_gen values
        if self.gen_keys is None:
            assert self.gen_data is None
            self.gen_keys = list(self._curr_gen_data.keys())
            self.gen_data = GrowableArray(len(self.gen_keys))

        values = np.full(len(self.gen_keys), np.NaN)
        for i, k in enumerate(self.gen_keys):
            values[i] = self._curr_gen_data.pop(k)
        self.gen_data.append(values)

        batch_len = len(self.batch_data) - len(self.batch_axis)

        # average the batch data and store it
        self.gen_average_data.append(np.mean(self.batch_data.values[-batch_len:, :], axis=0))

        # extend batch_axis array
        if batch_len > 0:
            for p in np.linspace(0, 1, num=batch_len, endpoint=False):
                self.batch_axis.append([len(self.gen_data) - 1 + p])

        # prepare for next gen
        assert not self._curr_gen_data
        self._curr_gen_data = None

    def finish_batch(self):
        assert self._curr_batch_data is not None, "No batch was started"

        if self.batch_keys is None:
            assert self.batch_data is None
            self.batch_keys = list(self._curr_batch_data.keys())
            self.batch_data = GrowableArray(len(self.batch_keys))
            self.gen_average_data = GrowableArray(len(self.batch_keys))

        values = np.full(len(self.batch_keys), np.NaN)
        for i, k in enumerate(self.batch_keys):
            values[i] = self._curr_batch_data.pop(k)
        self.batch_data.append(values)

        self._curr_batch_data = None

    def log_gen(self, ty: str, key: str, value):
        key = (ty, key)

        assert self._curr_gen_data is not None, "No gen has started yet"
        assert key not in self._curr_gen_data, f"Value for {key} already logged in this gen"
        if self.gen_keys is not None:
            assert key in self.gen_keys, f"Unexpected key {key}"

        self._curr_gen_data[key] = value

    def log_batch(self, ty: str, key: str, value):
        key = (ty, key)

        assert self._curr_batch_data is not None, "No batch has started yet"
        assert key not in self._curr_batch_data, f"Value for {key} already logged in this batch"
        if self.batch_keys is not None:
            assert key in self.batch_keys, f"Unexpected key {key}"

        self._curr_batch_data[key] = value


class GrowableArray:
    def __init__(self, width: int):
        self.width = width

        self._values = np.full((1, width), np.NaN)
        self._next_i = 0

    def __len__(self):
        return self._next_i

    @property
    def values(self):
        return self._values[:self._next_i, :]

    def append(self, values: np.array):
        assert len(values) == self.width

        # grow array if necessary
        if self._next_i == len(self._values):
            old_values = self._values
            self._values = np.full((2 * len(old_values), self.width), np.NaN)
            self._values[:len(old_values), :] = old_values

        # actually append values
        self._values[self._next_i, :] = values
        self._next_i += 1
