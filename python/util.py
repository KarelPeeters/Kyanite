import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {DEVICE}")

DATA_WIDTH = 3 + 1 + 4 * 81 + 2 * 9


class GenericData:
    def __init__(self, full):
        assert len(full.shape) == 2
        width = full.shape[1]
        assert width == DATA_WIDTH or width + 1 == DATA_WIDTH
        has_value = width == DATA_WIDTH

        self.full = full

        i = 0

        def take(length: int):
            nonlocal i
            i += length
            return full[:, i - length:i]

        self.y_win = take(3)
        self.y_final_value = self.y_win[:, 0] - self.y_win[:, 2]
        self.y_value = take(1) if has_value else None

        self.y_move_prob = take(81)
        self.mask_flat = take(81)
        self.x_tiles = take(2 * 81).view(-1, 2, 81)
        self.x_macros = take(2 * 9).view(-1, 2, 9)

        assert i == DATA_WIDTH or i + 1 == DATA_WIDTH

    def to(self, device):
        return GenericData(self.full.to(device))

    def pick_batch(self, indices):
        return GenericData(self.full[indices, :])

    def __len__(self):
        return len(self.full)


@dataclass
class GoogleData:
    input: Tensor
    value: Tensor
    final_value: Tensor
    policy: Tensor

    @staticmethod
    def from_generic(data: GenericData):
        o = o_tensor(data.full.device)
        input = torch.cat([
            data.mask_flat[:, o].view(-1, 1, 9, 9),
            data.x_tiles[:, :, o].view(-1, 2, 9, 9),
            data.x_macros.repeat_interleave(9, 2)[:, :, o].view(-1, 2, 9, 9)
        ], dim=1)

        value = data.y_value.view(-1, 1)
        final_value = data.y_final_value.view(-1, 1)
        # the shape of policy doesn't really matter, because the model ends in a dense layer anyway
        policy = data.y_move_prob.view(-1, 81)[:, o]
        return GoogleData(input=input, value=value, final_value=final_value, policy=policy)

    def to(self, device):
        return GoogleData(
            input=self.input.to(device),
            value=self.value.to(device),
            final_value=self.final_value.to(device),
            policy=self.policy.to(device)
        )

    def pick_batch(self, indices):
        return GoogleData(
            input=self.input[indices],
            value=self.value[indices],
            final_value=self.final_value[indices],
            policy=self.policy[indices],
        )

    @property
    def mask(self):
        return self.input[:, 0, :, :]

    def __len__(self):
        return len(self.input)


def load_data(path_csv) -> GenericData:
    path_tensor = Path(path_csv).with_suffix(".pt")

    if not os.path.exists(path_tensor) or os.path.getmtime(path_csv) > os.path.getmtime(path_tensor):
        print(f"Mapping data from {path_csv} to {path_tensor}")
        np_data = np.loadtxt(path_csv, delimiter=",", ndmin=2, dtype=np.float32)
        data = torch.tensor(np_data)
        torch.save(data, path_tensor)
    else:
        print(f"Using cached data {path_tensor}")
        data = torch.load(path_tensor)

    return GenericData(data)


def o_tensor(device):
    r = torch.arange(9, device=device)
    os_full = r.view(3, 3).repeat(3, 3)
    om_full = r.view(3, 3).repeat_interleave(3, 0).repeat_interleave(3, 1)
    o_full = (9 * om_full + os_full).view(81)
    return o_full


def linspace_int(stop: int, num: int) -> np.array:
    assert 0 < stop and 0 < num

    if num >= stop:
        return np.arange(stop)
    else:
        return np.linspace(0, stop - 1, num).astype(int)


def uniform_window_filter(data: np.array, window_size: int) -> np.array:
    if len(data) <= window_size * 3:
        return data

    from scipy.signal import filtfilt
    a = 1
    b = [1 / window_size] * window_size
    return filtfilt(b, a, data)
