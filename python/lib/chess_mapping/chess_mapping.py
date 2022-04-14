import os

import numpy as np
import torch

from lib.util import DEVICE


def rel_path(name: str):
    return os.path.join(os.path.dirname(__file__), name)


def load_file_lines_to_tensor(name: str):
    with open(rel_path(name)) as f:
        data = [int(line) for line in f]

    return torch.tensor(data, device=DEVICE, dtype=torch.int64)


CHESS_FLAT_TO_CONV = load_file_lines_to_tensor("chess_flat_to_conv.txt")
assert CHESS_FLAT_TO_CONV.shape == (1880,), CHESS_FLAT_TO_CONV.shape

CHESS_FLAT_TO_ATT = load_file_lines_to_tensor("chess_flat_to_att.txt")
assert CHESS_FLAT_TO_CONV.shape == (1880,), CHESS_FLAT_TO_CONV.shape

CHESS_FLAT_TO_MOVE_INPUT = np.genfromtxt(
    rel_path("chess_flat_to_move_input.txt"),
    delimiter=",", dtype=np.int32
)[:, :-1]
assert CHESS_FLAT_TO_MOVE_INPUT.shape == (1880, 8), CHESS_FLAT_TO_MOVE_INPUT.shape
