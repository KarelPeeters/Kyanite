import random
from math import prod
from pathlib import Path
from typing import Optional

import numpy
import numpy as np
import torch
from torch import nn
from torch.fx._experimental.fuser import fuse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {DEVICE}")

DATA_WIDTH = 3 + 3 + 17 * 7 * 7 + 17 * 7 * 7 + 3 * 7 * 7


class GameData:
    def __init__(self, full):
        assert len(full.shape) == 2
        assert full.shape[1] == DATA_WIDTH, f"Expected size {DATA_WIDTH}, got {full.shape[1]}"

        self.full = full

        i = 0

        def take(length: int):
            nonlocal i
            i += length
            return full[:, i - length:i]

        self.wdl_final = take(3)
        self.wdl_est = take(3)

        self.policy_mask = take(17 * 7 * 7).view(-1, 17, 7, 7)
        self.policy = take(17 * 7 * 7).view(-1, 17, 7, 7)

        self.board = take(3 * 7 * 7).view(-1, 3, 7, 7)

        assert i == DATA_WIDTH

    def to(self, device):
        return GameData(self.full.to(device))

    def random_symmetry(self):
        flip = bool(random.getrandbits(1))
        k = random.randrange(4)

        def transform(x):
            assert len(x.shape) > 2, x.shape
            assert x.shape[-1] == 7 and x.shape[-2] == 7, x.shape

            if flip:
                x = x.flip(-1)
            return x.rot90(k, [-1, -2])

        return GameData(
            torch.cat([
                self.wdl_final,
                self.wdl_est,
                transform(self.policy_mask).reshape(-1, 17 * 7 * 7),
                transform(self.policy).reshape(-1, 17 * 7 * 7),
                transform(self.board).reshape(-1, 3 * 7 * 7),
            ], dim=1)
        )

    def __getitem__(self, indices):
        return GameData(self.full[indices, :])

    def __len__(self):
        return len(self.full)


def load_data_multiple(paths: [str], test_fraction: float, limit_each: Optional[int] = None) -> (GameData, GameData):
    """
    All paths bust be .bin files
    This function does not actually shuffle train and test data itself, but games are randomly split between them.
    This is okay because they're shuffled during training anyway.
    """

    load_count = -1 if limit_each is None else limit_each * (1 + DATA_WIDTH)
    games = torch.zeros(0, 1 + DATA_WIDTH)

    for path in paths:
        path = Path(path)

        assert path.suffix == ".bin", f"Unexpected extension '{path.suffix}'"

        print(f"Loading data")
        part_games = numpy.fromfile(path, dtype=np.float32, count=load_count)
        if len(part_games) == 0:
            raise ValueError(f"Empty file {path}")

        part_games = torch.tensor(part_games).view(-1, 1 + DATA_WIDTH)
        games = torch.cat([games, part_games], dim=0)

    print("Splitting data")
    game_ids = games[:, 0].round().long()
    game_count = game_ids.max() + 1
    full = games[:, 1:]

    perm_games = torch.randperm(game_count)
    split_index = int((1 - test_fraction) * game_count)

    train_mask = perm_games[game_ids] < split_index
    train_data = GameData(full[train_mask, :])
    test_data = GameData(full[~train_mask, :])

    print(f"Train size {len(train_data)}, test size {len(test_data)}")
    return train_data, test_data


def load_data(path: str, test_fraction: float, limit: Optional[int]) -> (GameData, GameData):
    return load_data_multiple([path], test_fraction, limit)


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


def print_param_count(module: nn.Module, f=None):
    param_count = sum(prod(p.shape) for p in module.parameters())
    print(f"Model has {param_count} parameters, which takes {param_count // 1024 / 1024:.3f} Mb", file=f)
    for name, child in module.named_children():
        child_param_count = sum(prod(p.shape) for p in child.parameters())
        print(f"  {name}: {child_param_count / param_count:.2f}", file=f)


def save_fused_params(model, path):
    """Save the parameters of a fused version of the model as an .npz file."""
    model_fused = fuse(model)

    params = []
    for i, param in enumerate(model_fused.parameters()):
        if prod(param.shape) == 1:
            print(f"{i}: {param.data}")
        else:
            print(f"{i}: {param.shape}")
        params.append(param.detach().numpy())

    np.savez(path, *params)


SYMMETRY_INDICES_O = torch.tensor([
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
     31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
     60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
    [60, 61, 62, 57, 58, 59, 54, 55, 56, 69, 70, 71, 66, 67, 68, 63, 64, 65, 78, 79, 80, 75, 76, 77, 72, 73, 74, 33, 34,
     35, 30, 31, 32, 27, 28, 29, 42, 43, 44, 39, 40, 41, 36, 37, 38, 51, 52, 53, 48, 49, 50, 45, 46, 47, 6, 7, 8, 3, 4,
     5, 0, 1, 2, 15, 16, 17, 12, 13, 14, 9, 10, 11, 24, 25, 26, 21, 22, 23, 18, 19, 20],
    [20, 19, 18, 23, 22, 21, 26, 25, 24, 11, 10, 9, 14, 13, 12, 17, 16, 15, 2, 1, 0, 5, 4, 3, 8, 7, 6, 47, 46, 45, 50,
     49, 48, 53, 52, 51, 38, 37, 36, 41, 40, 39, 44, 43, 42, 29, 28, 27, 32, 31, 30, 35, 34, 33, 74, 73, 72, 77, 76, 75,
     80, 79, 78, 65, 64, 63, 68, 67, 66, 71, 70, 69, 56, 55, 54, 59, 58, 57, 62, 61, 60],
    [80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52,
     51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23,
     22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    [0, 3, 6, 1, 4, 7, 2, 5, 8, 27, 30, 33, 28, 31, 34, 29, 32, 35, 54, 57, 60, 55, 58, 61, 56, 59, 62, 9, 12, 15, 10,
     13, 16, 11, 14, 17, 36, 39, 42, 37, 40, 43, 38, 41, 44, 63, 66, 69, 64, 67, 70, 65, 68, 71, 18, 21, 24, 19, 22, 25,
     20, 23, 26, 45, 48, 51, 46, 49, 52, 47, 50, 53, 72, 75, 78, 73, 76, 79, 74, 77, 80],
    [60, 57, 54, 61, 58, 55, 62, 59, 56, 33, 30, 27, 34, 31, 28, 35, 32, 29, 6, 3, 0, 7, 4, 1, 8, 5, 2, 69, 66, 63, 70,
     67, 64, 71, 68, 65, 42, 39, 36, 43, 40, 37, 44, 41, 38, 15, 12, 9, 16, 13, 10, 17, 14, 11, 78, 75, 72, 79, 76, 73,
     80, 77, 74, 51, 48, 45, 52, 49, 46, 53, 50, 47, 24, 21, 18, 25, 22, 19, 26, 23, 20],
    [20, 23, 26, 19, 22, 25, 18, 21, 24, 47, 50, 53, 46, 49, 52, 45, 48, 51, 74, 77, 80, 73, 76, 79, 72, 75, 78, 11, 14,
     17, 10, 13, 16, 9, 12, 15, 38, 41, 44, 37, 40, 43, 36, 39, 42, 65, 68, 71, 64, 67, 70, 63, 66, 69, 2, 5, 8, 1, 4,
     7, 0, 3, 6, 29, 32, 35, 28, 31, 34, 27, 30, 33, 56, 59, 62, 55, 58, 61, 54, 57, 60],
    [80, 77, 74, 79, 76, 73, 78, 75, 72, 53, 50, 47, 52, 49, 46, 51, 48, 45, 26, 23, 20, 25, 22, 19, 24, 21, 18, 71, 68,
     65, 70, 67, 64, 69, 66, 63, 44, 41, 38, 43, 40, 37, 42, 39, 36, 17, 14, 11, 16, 13, 10, 15, 12, 9, 62, 59, 56, 61,
     58, 55, 60, 57, 54, 35, 32, 29, 34, 31, 28, 33, 30, 27, 8, 5, 2, 7, 4, 1, 6, 3, 0],
], dtype=torch.long, device=DEVICE)

SYMMETRY_INDICES_OO = torch.tensor([
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [6, 7, 8, 3, 4, 5, 0, 1, 2],
    [2, 1, 0, 5, 4, 3, 8, 7, 6],
    [8, 7, 6, 5, 4, 3, 2, 1, 0],
    [0, 3, 6, 1, 4, 7, 2, 5, 8],
    [6, 3, 0, 7, 4, 1, 8, 5, 2],
    [2, 5, 8, 1, 4, 7, 0, 3, 6],
    [8, 5, 2, 7, 4, 1, 6, 3, 0],
], dtype=torch.long, device=DEVICE)
