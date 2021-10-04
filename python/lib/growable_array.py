import numpy as np


class GrowableArray:
    def __init__(self, initial_values=None):
        if initial_values is None:
            self._values = np.full(1, np.NaN)
            self._next_i = 0
        else:
            assert len(initial_values.shape) == 1
            self._values = initial_values
            self._next_i = len(initial_values)

    @property
    def values(self):
        return self._values[:self._next_i]

    def __len__(self):
        return self._next_i

    def __getitem__(self, item):
        return self.values[item]

    def __setitem__(self, key, value):
        self.values[key] = value

    def _ensure_space(self, n: int):
        old_values = self._values

        if self._next_i + n > len(old_values):
            new_size = max(2 * len(old_values), self._next_i + n)

            self._values = np.full(new_size, np.NaN)
            self._values[:len(old_values)] = old_values

    def append(self, value):
        self._ensure_space(1)
        self._values[self._next_i] = value
        self._next_i += 1

    def extend(self, values: np.array):
        assert len(values.shape) == 1
        n = len(values)

        self._ensure_space(n)
        self._values[self._next_i:self._next_i + n, :] = values
        self._next_i += n
