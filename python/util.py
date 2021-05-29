import os
from pathlib import Path

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {DEVICE}")

DATA_WIDTH = 3 + 4 * 81 + 2 * 9


class Data:
    def __init__(self, full):
        assert len(full.shape) == 2
        assert full.shape[1] == DATA_WIDTH
        self.full = full

        i = 0

        def take(length: int):
            nonlocal i
            i += length
            return full[:, i - length:i]

        self.y_win = take(3)
        self.y_value = self.y_win[:, 0] - self.y_win[:, 2]

        self.y_move_prob = take(81)
        self.mask_flat = take(81)
        self.mask = self.mask_flat.view(-1, 9, 9)

        self.x_tiles = take(2 * 81).view(-1, 2, 9, 9)
        self.x_macros = take(2 * 9).view(-1, 2, 3, 3)

        assert i == DATA_WIDTH

    def to(self, device):
        return Data(self.full.to(device))

    def pick_batch(self, indices):
        return Data(self.full[indices, :])

    def __len__(self):
        return len(self.full)


def load_data(path_csv) -> Data:
    path_tensor = Path(path_csv).with_suffix(".pt")

    if not os.path.exists(path_tensor) or os.path.getmtime(path_csv) > os.path.getmtime(path_tensor):
        print(f"Mapping data from {path_csv} to {path_tensor}")
        np_data = np.loadtxt(path_csv, delimiter=",", ndmin=2, dtype=np.float32)
        data = torch.tensor(np_data)
        torch.save(data, path_tensor)
    else:
        print(f"Using cached data {path_tensor}")
        data = torch.load(path_tensor)

    return Data(data.to(DEVICE))
