import numpy as np


class GrowableArray:
    def __init__(self, width: int, initial_values=None):
        self.width = width

        if initial_values is None:
            self._values = np.full((1, width), np.NaN)
            self._next_i = 0
        else:
            assert initial_values.shape[1] == width
            self._values = initial_values
            self._next_i = len(initial_values)

    def __len__(self):
        return self._next_i

    @property
    def values(self):
        return self._values[:self._next_i, :]

    def ensure_space(self, n: int):
        old_values = self._values

        if self._next_i + n > len(old_values):
            new_size = max(2 * len(old_values), self._next_i + n)

            self._values = np.full((new_size, self.width), np.NaN)
            self._values[:len(old_values), :] = old_values

    def append(self, values: np.array):
        assert len(values) == self.width

        self.ensure_space(1)
        self._values[self._next_i, :] = values
        self._next_i += 1

    def extend(self, values: np.array):
        n, width = values.shape
        assert self.width == width, f"Expected width {self.width}, got {width}"

        self.ensure_space(n)
        self._values[self._next_i:self._next_i + n, :] = values
        self._next_i += n