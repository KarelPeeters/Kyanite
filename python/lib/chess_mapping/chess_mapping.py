import os

import torch

from lib.util import DEVICE


def load_file_lines_to_tensor(name: str):
    path = os.path.join(os.path.dirname(__file__), name)

    with open(path) as f:
        data = [int(line) for line in f]

    return torch.tensor(data, device=DEVICE, dtype=torch.int64)


CHESS_FLAT_TO_CONV = load_file_lines_to_tensor("chess_flat_to_conv.txt")
CHESS_FLAT_TO_ATT = load_file_lines_to_tensor("chess_flat_to_att.txt")
