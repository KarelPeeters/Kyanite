import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {DEVICE}")

DATA_WIDTH = 3 + 3 + 81 + (3 * 81 + 2 * 9)


class GenericData:
    def __init__(self, full):
        assert len(full.shape) == 2
        assert full.shape[1] == DATA_WIDTH

        self.full = full

        i = 0

        def take(length: int):
            nonlocal i
            i += length
            return full[:, i - length:i]

        self.wdl_final = take(3)
        self.wdl_est = take(3)
        self.policy_o = take(81)

        self.mask_o = take(81)
        self.tiles_o = take(2 * 81).view(-1, 2, 81)
        # for macros o and yx order are the same
        self.macros = take(2 * 9).view(-1, 2, 9)

        assert i == DATA_WIDTH

    def to(self, device):
        return GenericData(self.full.to(device))

    def pick_batch(self, indices):
        return GenericData(self.full[indices, :])

    def __len__(self):
        return len(self.full)


@dataclass
class GoogleData:
    input: Tensor
    wdl_final: Tensor
    wdl_est: Tensor
    policy: Tensor

    @staticmethod
    def from_generic(data: GenericData):
        o = o_tensor(data.full.device)
        input = torch.cat([
            data.mask_o[:, o].view(-1, 1, 9, 9),
            data.tiles_o[:, :, o].view(-1, 2, 9, 9),
            data.macros.repeat_interleave(9, 2)[:, :, o].view(-1, 2, 9, 9)
        ], dim=1)
        pred_policy = data.policy_o.view(-1, 81)[:, o].view(-1, 9, 9)

        return GoogleData(
            input=input,
            wdl_final=data.wdl_final,
            wdl_est=data.wdl_est,
            policy=pred_policy
        )

    def to(self, device):
        return GoogleData(
            input=self.input.to(device),
            wdl_final=self.wdl_final.to(device),
            wdl_est=self.wdl_est.to(device),
            policy=self.policy.to(device),
        )

    def pick_batch(self, indices):
        return GoogleData(
            input=self.input[indices],
            wdl_final=self.wdl_final[indices],
            wdl_est=self.wdl_est[indices],
            policy=self.policy[indices],
        )

    def random_symmetry(self):
        picked = torch.randint(0, 8, size=(len(self),))
        indices = SYMMETRY_INDICES_YX[picked]

        view = indices.view(-1, 1, 81).expand(-1, 5, -1)
        input_sym = torch.gather(self.input.view(-1, 5, 81), 2, view).view(-1, 5, 9, 9)

        policy_sym = torch.gather(self.policy.view(-1, 81), 1, indices).view(-1, 9, 9)

        return GoogleData(
            input=input_sym,
            wdl_final=self.wdl_final,
            wdl_est=self.wdl_est,
            policy=policy_sym,
        )

    @property
    def mask(self):
        return self.input[:, 0, :, :]

    def __len__(self):
        return len(self.input)


def load_data(path_csv, shuffle: bool) -> GenericData:
    path_tensor = Path(path_csv).with_suffix(".pt")

    if not os.path.exists(path_tensor) or os.path.getmtime(path_csv) > os.path.getmtime(path_tensor):
        print(f"Mapping data from {path_csv} to {path_tensor}")
        np_data = np.loadtxt(path_csv, delimiter=",", ndmin=2, dtype=np.float32)
        data = torch.tensor(np_data)
        torch.save(data, path_tensor)
    else:
        print(f"Using cached data {path_tensor}")
        data = torch.load(path_tensor)

    if shuffle:
        data = data[torch.randperm(len(data))]

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
    if len(data) <= window_size * 3 or window_size < 2:
        return data

    from scipy.signal import filtfilt
    a = 1
    b = [1 / window_size] * window_size
    return filtfilt(b, a, data)


SYMMETRY_INDICES_YX = torch.tensor([
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
     31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
     60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
    [72, 73, 74, 75, 76, 77, 78, 79, 80, 63, 64, 65, 66, 67, 68, 69, 70, 71, 54, 55, 56, 57, 58, 59, 60, 61, 62, 45, 46,
     47, 48, 49, 50, 51, 52, 53, 36, 37, 38, 39, 40, 41, 42, 43, 44, 27, 28, 29, 30, 31, 32, 33, 34, 35, 18, 19, 20, 21,
     22, 23, 24, 25, 26, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8],
    [8, 7, 6, 5, 4, 3, 2, 1, 0, 17, 16, 15, 14, 13, 12, 11, 10, 9, 26, 25, 24, 23, 22, 21, 20, 19, 18, 35, 34, 33, 32,
     31, 30, 29, 28, 27, 44, 43, 42, 41, 40, 39, 38, 37, 36, 53, 52, 51, 50, 49, 48, 47, 46, 45, 62, 61, 60, 59, 58, 57,
     56, 55, 54, 71, 70, 69, 68, 67, 66, 65, 64, 63, 80, 79, 78, 77, 76, 75, 74, 73, 72],
    [80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52,
     51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23,
     22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    [0, 9, 18, 27, 36, 45, 54, 63, 72, 1, 10, 19, 28, 37, 46, 55, 64, 73, 2, 11, 20, 29, 38, 47, 56, 65, 74, 3, 12, 21,
     30, 39, 48, 57, 66, 75, 4, 13, 22, 31, 40, 49, 58, 67, 76, 5, 14, 23, 32, 41, 50, 59, 68, 77, 6, 15, 24, 33, 42,
     51, 60, 69, 78, 7, 16, 25, 34, 43, 52, 61, 70, 79, 8, 17, 26, 35, 44, 53, 62, 71, 80],
    [72, 63, 54, 45, 36, 27, 18, 9, 0, 73, 64, 55, 46, 37, 28, 19, 10, 1, 74, 65, 56, 47, 38, 29, 20, 11, 2, 75, 66, 57,
     48, 39, 30, 21, 12, 3, 76, 67, 58, 49, 40, 31, 22, 13, 4, 77, 68, 59, 50, 41, 32, 23, 14, 5, 78, 69, 60, 51, 42,
     33, 24, 15, 6, 79, 70, 61, 52, 43, 34, 25, 16, 7, 80, 71, 62, 53, 44, 35, 26, 17, 8],
    [8, 17, 26, 35, 44, 53, 62, 71, 80, 7, 16, 25, 34, 43, 52, 61, 70, 79, 6, 15, 24, 33, 42, 51, 60, 69, 78, 5, 14, 23,
     32, 41, 50, 59, 68, 77, 4, 13, 22, 31, 40, 49, 58, 67, 76, 3, 12, 21, 30, 39, 48, 57, 66, 75, 2, 11, 20, 29, 38,
     47, 56, 65, 74, 1, 10, 19, 28, 37, 46, 55, 64, 73, 0, 9, 18, 27, 36, 45, 54, 63, 72],
    [80, 71, 62, 53, 44, 35, 26, 17, 8, 79, 70, 61, 52, 43, 34, 25, 16, 7, 78, 69, 60, 51, 42, 33, 24, 15, 6, 77, 68,
     59, 50, 41, 32, 23, 14, 5, 76, 67, 58, 49, 40, 31, 22, 13, 4, 75, 66, 57, 48, 39, 30, 21, 12, 3, 74, 65, 56, 47,
     38, 29, 20, 11, 2, 73, 64, 55, 46, 37, 28, 19, 10, 1, 72, 63, 54, 45, 36, 27, 18, 9, 0],
], dtype=torch.long, device=DEVICE)
